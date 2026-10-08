/** fast-formula-parser ships no types. Only the surface this project uses is
 *  declared — widening it later is cheaper than a wrong guess now. */
declare module "fast-formula-parser" {
  export interface Coord {
    row: number;
    col: number;
    sheet?: string;
  }
  export interface RangeRef {
    from: Coord;
    to: Coord;
    sheet?: string;
  }
  export type Ref = Coord | RangeRef;

  /** Thrown values are objects, not exceptions: `#DIV/0!` and friends travel
   *  through the evaluator as data so a bad cell does not take the sheet with
   *  it. `_error` is the text Excel prints. */
  export class FormulaError {
    _error: string;
  }

  /** Which cells a formula reads, without evaluating it. This is the whole
   *  dependency graph, handed over — the topological sort below stands on it. */
  export class DepParser {
    constructor(config?: unknown);
    parse(formula: string, position: Coord): Ref[];
  }

  export default class FormulaParser {
    constructor(config?: {
      onCell?: (ref: Coord) => number | string | boolean | null;
      onRange?: (ref: RangeRef) => (number | string | boolean | null)[][];
    });
    parse(formula: string, position: Coord, allowReturnArray?: boolean): unknown;
  }
}
