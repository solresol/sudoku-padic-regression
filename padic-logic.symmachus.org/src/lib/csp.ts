export type BooleanOp = "or" | "xor" | "and";

export interface Variable {
  name: string;
  index: number;
}

export type Expression =
  | { type: "literal"; name: string }
  | { type: "not"; expr: Expression }
  | { type: "binary"; op: BooleanOp; left: Expression; right: Expression };

export interface Constraint {
  id: number;
  source: string;
  expr: Expression;
}

export interface ParsedProblem {
  source: string;
  variables: Variable[];
  constraints: Constraint[];
}

export interface TernaryClause {
  id: number;
  op: "or";
  terms: Expression[];
  source: string;
  expr: Expression;
}

export interface CompiledProblem extends ParsedProblem {
  ternaryClauses: TernaryClause[];
  evaluatorSource: string;
  assignmentCount: number;
  validation: {
    maxClauseWidth: number;
    sourceClausesPreserved: boolean;
    readyForSearch: boolean;
  };
  scoring: {
    prime: number;
    power: number;
    theoreticalFloor: number;
    unitWellViolationsAllowed: number;
    nonUnitConstraintTarget: number;
  };
}

export interface AssignmentEvaluation {
  loss: number;
  theoreticalFloor: number;
  unitWellViolations: number;
  nonUnitSatisfied: number;
  totalConstraints: number;
}

type TokenKind =
  | "identifier"
  | "not"
  | "or"
  | "xor"
  | "and"
  | "leftParen"
  | "rightParen"
  | "eof";

interface Token {
  kind: TokenKind;
  value: string;
}

const OP_WORDS: Record<string, TokenKind> = {
  and: "and",
  or: "or",
  xor: "xor",
  not: "not"
};

export function parseProblem(source: string): ParsedProblem {
  const constraints: Constraint[] = [];
  const variables = new Map<string, Variable>();

  const lines = source
    .split(/\r?\n/u)
    .map((line) => line.replace(/#.*/u, "").trim())
    .filter(Boolean);

  for (const line of lines) {
    const expr = parseExpression(line);
    collectVariables(expr, variables);
    constraints.push({
      id: constraints.length + 1,
      source: line,
      expr
    });
  }

  return {
    source,
    variables: Array.from(variables.values()),
    constraints
  };
}

export function compileProblem(source: string): CompiledProblem {
  const parsed = parseProblem(source);
  const ternaryClauses = parsed.constraints
    .flatMap((constraint) => createCnfClauses(constraint))
    .map((clause, index) => ({ ...clause, id: index + 1 }));
  const maxClauseWidth = ternaryClauses.reduce(
    (maxWidth, clause) => Math.max(maxWidth, clause.terms.length),
    0
  );

  const compiled: CompiledProblem = {
    ...parsed,
    ternaryClauses,
    evaluatorSource: "",
    assignmentCount: 2 ** parsed.variables.length,
    validation: {
      maxClauseWidth,
      sourceClausesPreserved: parsed.constraints.length > 0,
      readyForSearch: parsed.constraints.length > 0 && maxClauseWidth <= 3
    },
    scoring: {
      prime: 17,
      power: 4,
      theoreticalFloor: 0,
      unitWellViolationsAllowed: 0,
      nonUnitConstraintTarget: 0
    }
  };

  return {
    ...compiled,
    evaluatorSource: buildEvaluatorSource(compiled)
  };
}

export function evaluateAssignment(
  compiled: CompiledProblem,
  assignment: Record<string, boolean>
): AssignmentEvaluation {
  let loss = 0;
  let nonUnitSatisfied = 0;

  for (const clause of compiled.ternaryClauses) {
    if (evaluateClause(clause, (name) => assignment[name] ?? false)) {
      nonUnitSatisfied += 1;
    } else {
      loss += 1;
    }
  }

  return {
    loss,
    theoreticalFloor: compiled.scoring.theoreticalFloor,
    unitWellViolations: 0,
    nonUnitSatisfied,
    totalConstraints: compiled.ternaryClauses.length
  };
}

export function renderClause(clause: TernaryClause): string {
  return `(${clause.terms.map(renderExpression).join(" v ")})`;
}

// The paper's Boolean clause reward. Encode truth by x_i = 0 (true) / 1 (false).
// A disjunctive clause is false iff every literal fails, i.e. iff u·x = t, where a
// positive literal z_i contributes +x_i (and +1 to t) and a negated literal ¬z_i
// contributes −x_i. So the clause is TRUE iff u·x ≠ t, and for a prime p > 3 the
// residual |u·x − t|_p is exactly the 0/1 indicator of clause satisfaction.
export interface ClauseAffineResidual {
  coeffs: Array<{ name: string; sign: 1 | -1 }>;
  t: number;
}

export function clauseAffineResidual(clause: TernaryClause): ClauseAffineResidual {
  const coeffs: Array<{ name: string; sign: 1 | -1 }> = [];
  let t = 0;
  for (const term of clause.terms) {
    if (term.type === "literal") {
      coeffs.push({ name: term.name, sign: 1 });
      t += 1;
    } else if (term.type === "not" && term.expr.type === "literal") {
      coeffs.push({ name: term.expr.name, sign: -1 });
    }
  }
  return { coeffs, t };
}

export function renderClauseAffine(clause: TernaryClause): string {
  const { coeffs, t } = clauseAffineResidual(clause);
  if (coeffs.length === 0) {
    return `0 ≠ ${t}`;
  }
  const lhs = coeffs
    .map((c, index) => {
      if (index === 0) {
        return c.sign < 0 ? `−${c.name}` : c.name;
      }
      return c.sign < 0 ? ` − ${c.name}` : ` + ${c.name}`;
    })
    .join("");
  return `${lhs} ≠ ${t}`;
}

// The genuine p-adic clause reward |u·x − t|_p (with x_i = 0 for true, 1 for false),
// which equals 1 exactly when the clause is satisfied and 0 otherwise for p > 3.
export function pAdicClauseReward(
  clause: TernaryClause,
  assignment: Record<string, boolean>,
  p = 17
): number {
  const { coeffs, t } = clauseAffineResidual(clause);
  let residual = -t;
  for (const c of coeffs) {
    const x = assignment[c.name] ? 0 : 1; // true -> 0, false -> 1
    residual += c.sign * x;
  }
  if (residual === 0) {
    return 0;
  }
  let m = Math.abs(residual);
  let k = 0;
  while (m % p === 0) {
    m /= p;
    k += 1;
  }
  return p ** -k;
}

export function renderExpression(expr: Expression): string {
  switch (expr.type) {
    case "literal":
      return expr.name;
    case "not":
      return `~${renderExpression(expr.expr)}`;
    case "binary":
      return `${renderExpression(expr.left)} ${expr.op === "or" ? "v" : expr.op} ${renderExpression(expr.right)}`;
  }
}

export function evaluateExpression(
  expr: Expression,
  readVariable: (name: string) => boolean
): boolean {
  switch (expr.type) {
    case "literal":
      return readVariable(expr.name);
    case "not":
      return !evaluateExpression(expr.expr, readVariable);
    case "binary": {
      const left = evaluateExpression(expr.left, readVariable);
      const right = evaluateExpression(expr.right, readVariable);
      if (expr.op === "or") {
        return left || right;
      }
      if (expr.op === "and") {
        return left && right;
      }
      return left !== right;
    }
  }
}

function evaluateClause(
  clause: TernaryClause,
  readVariable: (name: string) => boolean
): boolean {
  return clause.terms.some((term) => evaluateExpression(term, readVariable));
}

export function buildEvaluatorSource(compiled: Pick<CompiledProblem, "variables" | "ternaryClauses">): string {
  const variableLines = compiled.variables.map((variable) => {
    const bitExpression =
      variable.index < 31
        ? `((mask >>> ${variable.index}) & 1) === 1`
        : `(Math.floor(mask / 2 ** ${variable.index}) % 2) === 1`;
    return `  const v${variable.index} = ${bitExpression}; // ${variable.name}`;
  });

  const constraintLines = compiled.ternaryClauses.map((clause) => {
    const expression = clauseToCode(clause, compiled.variables);
    return `  if (!(${expression})) loss += 1; // compiled CNF clause ${clause.id}`;
  });

  return [
    "function evaluateMask(mask) {",
    "  let loss = 0;",
    ...variableLines,
    ...constraintLines,
    "  return loss;",
    "}"
  ].join("\n");
}

function parseExpression(source: string): Expression {
  const parser = new Parser(tokenize(source));
  return parser.parse();
}

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let index = 0;

  while (index < source.length) {
    const char = source[index];
    if (/\s/u.test(char)) {
      index += 1;
      continue;
    }

    if (char === "(") {
      tokens.push({ kind: "leftParen", value: char });
      index += 1;
      continue;
    }

    if (char === ")") {
      tokens.push({ kind: "rightParen", value: char });
      index += 1;
      continue;
    }

    if (char === "~" || char === "!") {
      tokens.push({ kind: "not", value: char });
      index += 1;
      continue;
    }

    if (char === "^") {
      tokens.push({ kind: "and", value: char });
      index += 1;
      continue;
    }

    if (char === "⊕") {
      tokens.push({ kind: "xor", value: char });
      index += 1;
      continue;
    }

    if (char === "&" && source[index + 1] === "&") {
      tokens.push({ kind: "and", value: "&&" });
      index += 2;
      continue;
    }

    if (char === "|" && source[index + 1] === "|") {
      tokens.push({ kind: "or", value: "||" });
      index += 2;
      continue;
    }

    const wordMatch = /^[A-Za-z_][A-Za-z0-9_]*/u.exec(source.slice(index));
    if (wordMatch) {
      const value = wordMatch[0];
      const lower = value.toLowerCase();
      const operator = value === "v" ? "or" : OP_WORDS[lower];
      tokens.push({
        kind: operator ?? "identifier",
        value
      });
      index += value.length;
      continue;
    }

    throw new Error(`Unexpected token "${char}" in "${source}".`);
  }

  tokens.push({ kind: "eof", value: "" });
  return tokens;
}

class Parser {
  private cursor = 0;

  constructor(private readonly tokens: Token[]) {}

  parse(): Expression {
    const expr = this.parseOr();
    this.expect("eof");
    return expr;
  }

  private parseOr(): Expression {
    let expr = this.parseXor();
    while (this.match("or")) {
      expr = { type: "binary", op: "or", left: expr, right: this.parseXor() };
    }
    return expr;
  }

  private parseXor(): Expression {
    let expr = this.parseAnd();
    while (this.match("xor")) {
      expr = { type: "binary", op: "xor", left: expr, right: this.parseAnd() };
    }
    return expr;
  }

  private parseAnd(): Expression {
    let expr = this.parseUnary();
    while (this.match("and")) {
      expr = { type: "binary", op: "and", left: expr, right: this.parseUnary() };
    }
    return expr;
  }

  private parseUnary(): Expression {
    if (this.match("not")) {
      return { type: "not", expr: this.parseUnary() };
    }

    if (this.match("leftParen")) {
      const expr = this.parseOr();
      this.expect("rightParen");
      return expr;
    }

    const token = this.expect("identifier");
    return { type: "literal", name: token.value };
  }

  private match(kind: TokenKind): boolean {
    if (this.tokens[this.cursor].kind !== kind) {
      return false;
    }
    this.cursor += 1;
    return true;
  }

  private expect(kind: TokenKind): Token {
    const token = this.tokens[this.cursor];
    if (token.kind !== kind) {
      throw new Error(`Expected ${kind}, got ${token.value || token.kind}.`);
    }
    this.cursor += 1;
    return token;
  }
}

function collectVariables(expr: Expression, variables: Map<string, Variable>): void {
  switch (expr.type) {
    case "literal":
      if (!variables.has(expr.name)) {
        variables.set(expr.name, {
          name: expr.name,
          index: variables.size
        });
      }
      return;
    case "not":
      collectVariables(expr.expr, variables);
      return;
    case "binary":
      collectVariables(expr.left, variables);
      collectVariables(expr.right, variables);
      return;
  }
}

function createCnfClauses(constraint: Constraint): TernaryClause[] {
  return expressionToCnfTerms(constraint.expr).map((terms, index) => ({
    id: index + 1,
    op: "or",
    terms,
    source:
      index === 0
        ? constraint.source
        : `${constraint.source} [CNF expansion ${index + 1}]`,
    expr: constraint.expr
  }));
}

function expressionToCnfTerms(expr: Expression): Expression[][] {
  if (expr.type === "binary" && expr.op === "and") {
    return [
      ...expressionToCnfTerms(expr.left),
      ...expressionToCnfTerms(expr.right)
    ];
  }

  const orTerms = flattenSameOp(expr, "or");
  if (orTerms && orTerms.every(isLiteralExpression)) {
    return [orTerms];
  }

  return truthTableCnf(expr);
}

function flattenSameOp(expr: Expression, op: "or" | "xor"): Expression[] | null {
  if (expr.type === "binary" && expr.op === op) {
    const left = flattenSameOp(expr.left, op);
    const right = flattenSameOp(expr.right, op);
    if (!left || !right) {
      return null;
    }
    return [...left, ...right];
  }

  if (isLiteralExpression(expr)) {
    return [expr];
  }

  return null;
}

function truthTableCnf(expr: Expression): Expression[][] {
  const variableNames = Array.from(expressionVariableNames(expr));
  const clauses: Expression[][] = [];
  const assignmentCount = 2 ** variableNames.length;

  for (let mask = 0; mask < assignmentCount; mask += 1) {
    const assignment = new Map<string, boolean>();
    variableNames.forEach((name, index) => {
      assignment.set(name, ((mask >>> index) & 1) === 1);
    });

    const satisfied = evaluateExpression(expr, (name) => assignment.get(name) ?? false);
    if (!satisfied) {
      clauses.push(
        variableNames.map((name) =>
          assignment.get(name)
            ? ({ type: "not", expr: { type: "literal", name } } satisfies Expression)
            : ({ type: "literal", name } satisfies Expression)
        )
      );
    }
  }

  if (clauses.length === 0) {
    return [];
  }

  return clauses;
}

function expressionVariableNames(expr: Expression, names = new Set<string>()): Set<string> {
  switch (expr.type) {
    case "literal":
      names.add(expr.name);
      return names;
    case "not":
      return expressionVariableNames(expr.expr, names);
    case "binary":
      expressionVariableNames(expr.left, names);
      expressionVariableNames(expr.right, names);
      return names;
  }
}

function isLiteralExpression(expr: Expression): boolean {
  return (
    expr.type === "literal" ||
    (expr.type === "not" && expr.expr.type === "literal")
  );
}

function clauseToCode(clause: TernaryClause, variables: Variable[]): string {
  return clause.terms.map((term) => expressionToCode(term, variables)).join(" || ");
}

function expressionToCode(expr: Expression, variables: Variable[]): string {
  switch (expr.type) {
    case "literal": {
      const variable = variables.find((candidate) => candidate.name === expr.name);
      if (!variable) {
        throw new Error(`Unknown variable ${expr.name}`);
      }
      return `v${variable.index}`;
    }
    case "not":
      return `!(${expressionToCode(expr.expr, variables)})`;
    case "binary": {
      const left = expressionToCode(expr.left, variables);
      const right = expressionToCode(expr.right, variables);
      if (expr.op === "or") {
        return `(${left} || ${right})`;
      }
      if (expr.op === "and") {
        return `(${left} && ${right})`;
      }
      return `(${left} !== ${right})`;
    }
  }
}
