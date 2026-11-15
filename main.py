from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field, asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

_VENDORED_DEPENDENCIES = Path(__file__).with_name("vendor")
if _VENDORED_DEPENDENCIES.is_dir():
    sys.path.insert(0, str(_VENDORED_DEPENDENCIES))

import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Where
from sqlparse.tokens import Keyword, Whitespace, Punctuation

def normalized_sql(sql: str) -> str:
    return " ".join(sqlparse.format(sql, strip_comments=True).split()).casefold()

@dataclass(frozen=True)
class Attribute:
    name: str
    distinct: int
    unique: bool = False

@dataclass(frozen=True)
class Index:
    attributes: tuple[str, ...]
    kind: str
    clustered: bool
    height: int | None = None

@dataclass
class Table:
    name: str
    rows: float
    blocks: float
    rows_per_block: float
    attributes: dict[str, Attribute]
    indexes: list[Index]

    def attribute(self, column: str) -> Attribute:
        for name, attribute in self.attributes.items():
            if name.lower() == column.lower():
                return attribute
        return Attribute(column, max(1, int(self.rows)), False)

@dataclass
class Predicate:
    left: str
    operator: str
    right: str

    @property
    def is_join(self) -> bool:
        return "." in self.left and "." in self.right

@dataclass
class Query:
    select_items: list[str]
    relations: dict[str, str]
    predicates: list[Predicate]
    order_by: str | None

@dataclass
class Plan:
    operator: str
    algorithm: str
    rows: float
    blocks: float
    local_cost: float
    total_cost: float
    local_seeks: float
    total_seeks: float
    detail: str
    children: list["Plan"] = field(default_factory=list)
    relations: frozenset[str] = field(default_factory=frozenset)

def load_catalogue(document: dict[str, Any]) -> tuple[dict[str, Table], int]:
    table_records = document["schema"]["tables"]
    tables: dict[str, Table] = {}
    for raw in table_records:
        name = str(raw["name"])
        rows = float(raw["rowCount"])
        blocks = float(raw["blockCount"])
        rpb = float(raw["rowsPerBlock"])
        attrs: dict[str, Attribute] = {}
        for item in raw["attributes"]:
            attr_name = str(item["name"])
            attrs[attr_name] = Attribute(
                attr_name,
                int(item["distinctValues"]),
                bool(item["unique"]),
            )
        indexes = []
        for item in raw["indexes"]:
            indexes.append(Index(
                tuple(map(str, item["attributes"])), str(item["type"]).lower(),
                bool(item["clustered"]), item.get("treeHeight"),
            ))
        tables[name.lower()] = Table(name, rows, blocks, rpb, attrs, indexes)
    return tables, max(3, int(document["bufferBlocks"]))

def identifiers(token: Any) -> list[str]:
    if isinstance(token, IdentifierList):
        return [str(item).strip() for item in token.get_identifiers()]
    if isinstance(token, Identifier):
        return [str(token).strip()]
    return [part.strip() for part in str(token).split(",") if part.strip()]

def parse_relation(text: str) -> tuple[str, str]:
    parts = text.strip().split()
    table = parts[0].strip('"`[]')
    alias = table
    if len(parts) >= 3 and parts[-2].upper() == "AS":
        alias = parts[-1]
    elif len(parts) >= 2:
        alias = parts[-1]
    return alias.strip('"`[]'), table

_PREDICATE = re.compile(r"^\s*([\w.\[\]`\"]+)\s*(=|<>|!=|<=|>=|<|>)\s*(.*?)\s*$", re.S)

def parse_predicates(where_text: str) -> list[Predicate]:
    body = re.sub(r"^\s*WHERE\s+", "", where_text, flags=re.I).strip()
    pieces = re.split(r"\s+AND\s+(?=(?:[^']*'[^']*')*[^']*$)", body, flags=re.I)
    result = []
    for piece in pieces:
        matched = _PREDICATE.match(piece)
        if matched:
            result.append(Predicate(*[part.strip(' "`[]') for part in matched.groups()]))
    return result

def parse_query(sql: str) -> Query:
    statement = sqlparse.parse(sql)[0]
    tokens = [t for t in statement.tokens if t.ttype not in Whitespace and t.ttype is not Punctuation]
    select_items: list[str] = []
    relations: dict[str, str] = {}
    order_by: str | None = None
    mode = ""
    for token in tokens:
        normalized = token.normalized.upper() if getattr(token, "normalized", None) else str(token).upper()
        if normalized == "SELECT":
            mode = "select"; continue
        if normalized == "FROM":
            mode = "from"; continue
        if isinstance(token, Where):
            predicates = parse_predicates(str(token)); mode = ""; continue
        if normalized.startswith("ORDER BY"):
            mode = "order"; continue
        if mode == "select":
            select_items.extend(identifiers(token))
        elif mode == "from":
            for relation in identifiers(token):
                alias, table = parse_relation(relation)
                relations[alias.lower()] = table
        elif mode == "order":
            order_by = str(token).split()[0].strip('"`[]')
            mode = ""
    return Query(select_items, relations, predicates if 'predicates' in locals() else [], order_by)

class Optimizer:
    def __init__(self, tables: dict[str, Table], buffer_size: int) -> None:
        self.tables, self.m = tables, buffer_size

    def table_for_alias(self, query: Query, alias: str) -> Table:
        return self.tables[query.relations[alias.lower()].lower()]

    @staticmethod
    def split_column(expression: str) -> tuple[str | None, str]:
        clean = expression.strip(' "`[]')
        return tuple(clean.rsplit(".", 1)) if "." in clean else (None, clean)

    def matching_index(self, table: Table, column: str) -> Index | None:
        for index in table.indexes:
            if index.attributes and index.attributes[0].lower() == column.lower():
                return index
        return None

    def clustered_order_index(self, query: Query, order_by: str, relations: frozenset[str]) -> Index | None:
        alias, column = self.split_column(order_by)
        if alias is None:
            if len(relations) != 1:
                return None
            alias = next(iter(relations))
        if alias.lower() not in relations or len(relations) != 1:
            return None
        table = self.table_for_alias(query, alias)
        for index in table.indexes:
            if (index.clustered and index.attributes and
                    index.attributes[0].lower() == column.lower() and
                    ("b" in index.kind or "tree" in index.kind)):
                return index
        return None

    def predicate_belongs_to_alias(self, query: Query, predicate: Predicate, alias: str) -> bool:
        qualifier, column = self.split_column(predicate.left)
        if qualifier is not None:
            return qualifier.lower() == alias.lower()
        if len(query.relations) == 1:
            return True
        owners = [candidate for candidate in query.relations
                  if any(name.lower() == column.lower()
                         for name in self.table_for_alias(query, candidate).attributes)]
        return len(owners) == 1 and owners[0].lower() == alias.lower()

    def index_height(self, index: Index, table: Table) -> int:
        return int(index.height) if index.height else max(1, math.ceil(math.log(max(1, table.blocks), 100)))

    def btree_probe_cost(self, index: Index, table: Table) -> int:
        return self.index_height(index, table) + 1

    def predicate_selectivity(self, table: Table, predicate: Predicate) -> float:
        _, column = self.split_column(predicate.left)
        attribute = table.attribute(column)
        if predicate.operator == "=":
            return 1 / max(1, table.rows if attribute.unique else attribute.distinct)
        return 1/2

    def external_sort_cost(self, blocks: float) -> float:
        if blocks <= self.m:
            return blocks
        runs = math.ceil(blocks / self.m)
        fan_in = max(2, self.m - 1)
        merge_passes = math.ceil(math.log(runs, fan_in))
        return blocks * (2 * merge_passes + 1)

    def external_sort_seeks(self, blocks: float) -> float:
        if blocks <= self.m:
            return 1
        runs = math.ceil(blocks / self.m)
        fan_in = max(2, self.m - 1)
        merge_passes = math.ceil(math.log(runs, fan_in))
        return 2 * runs + math.ceil(blocks) * (2 * merge_passes - 1)

    def local_access_plan(self, query: Query, alias: str) -> Plan:
        table = self.table_for_alias(query, alias)
        local = [p for p in query.predicates
                 if not p.is_join and self.predicate_belongs_to_alias(query, p, alias)]
        selectivity = math.prod(self.predicate_selectivity(table, p) for p in local) if local else 1.0
        rows = max(1.0, math.ceil(table.rows * selectivity))
        blocks = max(1.0, math.ceil(rows / table.rows_per_block))
        scan_output_cost = blocks if local else 0
        scan_output_seeks = 1 if local else 0
        candidates: list[tuple[float, float, str, str]] = [
            (table.blocks + scan_output_cost, 1 + scan_output_seeks, "A1 linear scan",
             "Read every table block" + (" and materialize matching blocks." if local else "."))
        ]
        for predicate in local:
            _, column = self.split_column(predicate.left)
            index = self.matching_index(table, column)
            if index is None:
                continue
            expected_rows = max(1.0, math.ceil(table.rows * self.predicate_selectivity(table, predicate)))
            expected_blocks = max(1.0, math.ceil(expected_rows / table.rows_per_block))
            if "hash" in index.kind and predicate.operator == "=":
                cost = 1 + (expected_blocks if index.clustered else expected_rows)
                candidates.append((cost, 1 + expected_rows, "hash index scan", f"Hash index on {column}."))
            elif "b" in index.kind or "tree" in index.kind:
                height = self.index_height(index, table)
                probe_cost = self.btree_probe_cost(index, table)
                if predicate.operator == "=":
                    fetch = expected_blocks if index.clustered else expected_rows
                    candidates.append((probe_cost + fetch, probe_cost + expected_rows, "B+ tree index scan", f"B+ tree on {column}, height {height}."))
                else:
                    fetch = expected_blocks if index.clustered else expected_rows
                    candidates.append((probe_cost + fetch, probe_cost + expected_rows, "B+ tree range scan", f"B+ tree range on {column}, height {height}."))
        equality_columns = {
            self.split_column(predicate.left)[1].lower()
            for predicate in local if predicate.operator == "="
        }
        for index in table.indexes:
            indexed_columns = [column.lower() for column in index.attributes]
            if (len(indexed_columns) > 1 and "hash" in index.kind and
                    all(column in equality_columns for column in indexed_columns)):
                fetch = blocks if index.clustered else rows
                candidates.append((1 + fetch + blocks, 1 + rows + 1,
                                   "composite hash index scan",
                                   "Hash index on (" + ", ".join(index.attributes) + ")."))
        cost, seeks, algorithm, detail = min(candidates, key=lambda item: item[0])
        suffix = "; ".join(f"{p.left} {p.operator} {p.right}" for p in local) or "no pushed predicate"
        return Plan("selection/access", algorithm, rows, blocks, cost, cost, seeks, seeks,
                    f"{detail} Conditions: {suffix}", [], frozenset({alias.lower()}))

    def join_predicates(self, query: Query, left: frozenset[str], right: frozenset[str]) -> list[Predicate]:
        found = []
        for predicate in query.predicates:
            if not predicate.is_join:
                continue
            a, _ = self.split_column(predicate.left); b, _ = self.split_column(predicate.right)
            if a and b and ((a.lower() in left and b.lower() in right) or (b.lower() in left and a.lower() in right)):
                found.append(predicate)
        return found

    def estimate_join_result(self, query: Query, left: Plan, right: Plan, conditions: list[Predicate]) -> tuple[float, float]:
        selectivity = 1.0
        for condition in conditions:
            a_alias, a_col = self.split_column(condition.left); b_alias, b_col = self.split_column(condition.right)
            if condition.operator == "=":
                va = self.table_for_alias(query, a_alias or "").attribute(a_col).distinct
                vb = self.table_for_alias(query, b_alias or "").attribute(b_col).distinct
                selectivity *= 1 / max(1, va, vb)
            else:
                selectivity *= 1 / 3
        rows = max(1.0, math.ceil(left.rows * right.rows * selectivity))
        density = max(1.0, min(left.rows / left.blocks, right.rows / right.blocks) / 2)
        return rows, max(1.0, math.ceil(rows / density))

    def join_candidates(self, query: Query, left: Plan, right: Plan, conditions: list[Predicate]) -> Iterable[tuple[str, float, float, str]]:
        lb, rb = left.blocks, right.blocks
        outer_chunks = math.ceil(lb / (self.m - 2))
        yield "block nested-loop join", lb + outer_chunks * rb, 2 * outer_chunks, f"Outer buffer has {self.m - 2} blocks."
        yield "tuple nested-loop join", lb + left.rows * rb, lb + left.rows, "Scans the right input once per left tuple."
        if all(c.operator == "=" for c in conditions):
            yield "sort-merge join", lb + rb + self.external_sort_cost(lb) + self.external_sort_cost(rb), self.external_sort_seeks(lb) + self.external_sort_seeks(rb) + 2, "Sort both inputs then merge once."
            build, probe = (left, right) if left.blocks <= right.blocks else (right, left)
            if build.blocks <= self.m - 1:
                yield "in-memory hash join", lb + rb, 2, "Build the smaller input in memory, then probe once."
            else:
                buffer_blocks = max(1, self.m // 3)
                yield "grace hash join", 3 * (lb + rb), 2 * (math.ceil(lb / buffer_blocks) + math.ceil(rb / buffer_blocks)), "Partition both inputs, then build/probe each partition."
            for outer, inner in ((left, right), (right, left)):
                if len(inner.relations) != 1:
                    continue
                inner_alias = next(iter(inner.relations))
                for condition in conditions:
                    a, a_col = self.split_column(condition.left); b, b_col = self.split_column(condition.right)
                    inner_column = a_col if a and a.lower() == inner_alias else b_col
                    index = self.matching_index(self.table_for_alias(query, inner_alias), inner_column)
                    if index:
                        lookup = (1 if "hash" in index.kind else self.btree_probe_cost(index, self.table_for_alias(query, inner_alias))) + 1
                        yield "indexed nested-loop join", outer.blocks + outer.rows * lookup, 1 + outer.rows * lookup, f"Use {index.kind} index on inner {inner_alias}.{inner_column}."
                        break

    def join(self, query: Query, left: Plan, right: Plan, conditions: list[Predicate]) -> Plan:
        if not conditions:
            conditions = []
        rows, blocks = self.estimate_join_result(query, left, right, conditions)
        algorithm, local_cost, local_seeks, detail = min(self.join_candidates(query, left, right, conditions), key=lambda item: item[1])
        return Plan("join", algorithm, rows, blocks, local_cost, left.total_cost + right.total_cost + local_cost,
                    local_seeks, left.total_seeks + right.total_seeks + local_seeks,
                    detail + (" Conditions: " + " AND ".join(f"{p.left}{p.operator}{p.right}" for p in conditions) if conditions else " Cartesian product."),
                    [left, right], left.relations | right.relations)

    def optimize(self, query: Query) -> Plan:
        aliases = tuple(query.relations)
        best: dict[frozenset[str], Plan] = {frozenset({a}): self.local_access_plan(query, a) for a in aliases}
        for size in range(2, len(aliases) + 1):
            for group in combinations(aliases, size):
                subset = frozenset(group)
                candidates = []
                for left_size in range(1, size):
                    for left_tuple in combinations(group, left_size):
                        left_set = frozenset(left_tuple); right_set = subset - left_set
                        if min(left_set) > min(right_set):
                            continue
                        candidates.append(self.join(query, best[left_set], best[right_set], self.join_predicates(query, left_set, right_set)))
                best[subset] = min(candidates, key=lambda candidate: candidate.total_cost)
        plan = best[frozenset(aliases)]
        plan = Plan("projection", "streaming projection", plan.rows, plan.blocks, 0, plan.total_cost, 0, plan.total_seeks,
                    "Keep: " + ", ".join(query.select_items), [plan], plan.relations)
        if query.order_by:
            ordered_index = self.clustered_order_index(query, query.order_by, plan.relations)
            if ordered_index:
                sort_cost = plan.blocks
                sort_seeks = 1
                sort_algorithm = "clustered B+ tree ordered scan"
                sort_detail = f"Read clustered index order by {query.order_by}."
            else:
                sort_cost = self.external_sort_cost(plan.blocks)
                sort_seeks = self.external_sort_seeks(plan.blocks)
                sort_algorithm = "external sort-merge"
                sort_detail = f"Order result by {query.order_by}."
            plan = Plan("sort", sort_algorithm, plan.rows, plan.blocks, sort_cost, plan.total_cost + sort_cost,
                        sort_seeks, plan.total_seeks + sort_seeks,
                        sort_detail, [plan], plan.relations)
        return plan


def rounded(value: float) -> int | float:
    return int(value) if float(value).is_integer() else round(value, 2)

def print_plan(plan: Plan, depth: int = 0) -> None:
    for child in plan.children:
        print_plan(child, depth + 1)
    prefix = "  " * depth
    print(f"{prefix}{plan.operator}: {plan.algorithm}")
    print(f"{prefix}  rows={rounded(plan.rows)}, blocks={rounded(plan.blocks)}, transfers={rounded(plan.local_cost)}, seeks={rounded(plan.local_seeks)}")
    print(f"{prefix}  {plan.detail}")

def plan_to_dict(plan: Plan) -> dict[str, Any]:
    data = asdict(plan)
    data["relations"] = sorted(plan.relations)
    data["rows"], data["blocks"] = rounded(plan.rows), rounded(plan.blocks)
    data["local_cost"], data["total_cost"] = rounded(plan.local_cost), rounded(plan.total_cost)
    data["local_seeks"], data["total_seeks"] = rounded(plan.local_seeks), rounded(plan.total_seeks)
    data["children"] = [plan_to_dict(child) for child in plan.children]
    return data

def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate an SQL SELECT evaluation plan.")
    parser.add_argument("input", type=Path,
                        help="Path to the course JSON database-metrics file")
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    tables, buffer_size = load_catalogue(document)
    optimizer = Optimizer(tables, buffer_size)
    while True:
        console_query = input("\nEnter an SQL SELECT query (or EXIT): ").strip()
        if console_query.upper() in {"EXIT", "QUIT"}:
            break
        if not console_query:
            print("No query entered.")
            continue
        print(f"\n=== Query (buffer={buffer_size} blocks) ===\n{console_query}")
        plan = optimizer.optimize(parse_query(console_query))
        print("\nPhysical plan (children first):")
        print_plan(plan)

        print(f"\nblock transfers: {rounded(plan.total_cost)}, seek cost: {rounded(plan.total_seeks)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
