import { connection } from "next/server";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tier } from "@/components/tier";
import { TIER, type Tier as TierKey } from "@/lib/report";
import {
  isSynthetic,
  METRIC_NAME,
  NEVER_MEASURED,
  readReport,
  REPORT_REL,
  RETRACTED,
  type MetricResult,
  type ValidationReport,
} from "@/lib/validation";

import { RunPanel } from "./run-panel";

/** The integrity page.
 *
 *  It answers one question in ten seconds — what may be said out loud — and
 *  then shows every number behind that answer, including the ones that make it
 *  look worse. A report that shows five confident numbers and no error bars is
 *  not more accurate than this one; it is the same measurement problem with the
 *  evidence removed.
 */
export default async function ChecksPage() {
  // The report is read from the filesystem, so prerendering would freeze one
  // build's copy of it into the page. This has to be the file as it is now.
  // Exception: the desktop export (NEXT_PUBLIC_FUTTRAFFIC_DESKTOP) IS a static
  // prerender by construction — the report freezes at packaging time, which is
  // correct for an artifact that ships its evidence with the build.
  if (process.env.NEXT_PUBLIC_FUTTRAFFIC_DESKTOP !== "1") await connection();
  const report = await readReport();

  return (
    <div className="mx-auto grid w-full max-w-5xl min-w-0 gap-4">
      <div>
        <h1 className="text-[18px] font-semibold tracking-[-0.2px]">Accuracy</h1>
        <p className="text-muted-foreground text-sm">
          What has been measured, what missed, and what has never been tested at all.
        </p>
      </div>

      {report ? <Verdict report={report.data} rel={report.rel} /> : <NoReport />}

      <TierLegend />
      <ModelTransfer />
      <NotEstablished />

      <RunPanel />

      {report && <Detail report={report.data} rel={report.rel} />}

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Retracted figures</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3">
          {RETRACTED.map(([claim, why]) => (
            <div key={claim} className="text-sm">
              <div className="text-muted-foreground line-through">{claim}</div>
              <p className="text-muted-foreground mt-1 text-xs">{why}</p>
            </div>
          ))}
          <p className="text-muted-foreground text-xs">
            Kept here so they cannot quietly return. The reports themselves live in{" "}
            <code className="bg-muted rounded px-1 py-0.5">model/validation/retired/</code>, which
            this page will not read.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

// ------------------------------------------------------------ trust tiers

/** The badge on every figure in the report says "/checks carries the long
 *  version". This is that long version, and it is the only place all three
 *  tiers are readable side by side. Wording comes from lib/report, so the badge
 *  and the explanation cannot drift apart. */
function TierLegend() {
  return (
    <Card className="gap-3">
      <CardHeader>
        <CardTitle className="text-sm">
          What the five badges on the report mean
        </CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {(["ok", "rel", "prov", "not_evaluated", "withheld"] as TierKey[]).map((t) => (
          <div key={t} className="grid gap-1.5">
            <div>
              <Tier t={t} />
            </div>
            <p className="text-muted-foreground text-[13px] leading-snug">{TIER[t].why}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

// -------------------------------------------------- model transfer evidence

/** PathCRF (KDD 2026, arXiv 2602.12080, MPL-2.0): the authors' released
 *  Bundesliga weights, run on our match with no fine-tuning.
 *
 *  These are MEASURED and STATIC. They describe one fixed experiment, so they
 *  are deliberately not wired to the API — an evidence page whose numbers move
 *  when a job finishes is not evidence. Nothing here is derived, averaged or
 *  rounded on the way to the screen; the strings are the results.
 */
const TRANSFER: { label: string; ours: string; paper: string }[] = [
  { label: "Event detection F1", ours: "0.6007", paper: "0.7569" },
  { label: "Sender naming", ours: "86.2%", paper: "80.44%" },
  { label: "Receiver naming", ours: "76.3%", paper: "82.28%" },
  { label: "Possession share error", ours: "−0.48 pp", paper: "+0.51 pp" },
];

/** No counterpart in the paper, so no comparison column: an empty cell reads as
 *  a missing result rather than as a question the paper does not answer. */
const TRANSFER_ONLY: { n: string; label: string; sub: string }[] = [
  { n: "1.77 m", label: "Possession location", sub: "median error" },
  { n: "0.40 s", label: "Event timing", sub: "median error" },
  { n: "21 s", label: "Inference cost", sub: "per half, on CPU" },
];

/** The whole page in three rows. Events are read off positions, so positional
 *  error is the input everything else is downstream of. */
const SENSITIVITY: [string, string][] = [
  ["0.0 m", "0.60"],
  ["0.5 m", "0.56"],
  ["1.0 m", "0.46"],
];

function ModelTransfer() {
  return (
    <Card className="border-t-ok min-w-0 gap-4 border-t-3">
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
          Model transfer
          <Badge variant="outline" className="text-ok">
            measured
          </Badge>
        </CardTitle>
      </CardHeader>

      <CardContent className="grid min-w-0 gap-5">
        <p className="text-[13px] leading-relaxed">
          PathCRF (KDD 2026, <span className="readout">arXiv 2602.12080</span>, MPL-2.0). The
          authors&apos; released Bundesliga weights, run on our match with{" "}
          <b>no fine-tuning</b> — nothing here was trained on our data. So these are transfer
          figures: what a published model does when it is handed a different competition. They
          say something about the model. They do not say anything about this pipeline
          end to end; the card below that one is the list of what they leave open.
        </p>

        <div className="grid min-w-0 gap-2">
          <h3 className="text-sm font-medium">
            Against the paper&apos;s own held-out result
          </h3>
          <Table>
            <TableHeader>
              <TableRow>
                {/* Short headers on purpose: the h3 above already says which
                    figure is which, and three long ones push this table into a
                    sideways scroll on a phone. */}
                <TableHead>Quantity</TableHead>
                <TableHead className="text-right">Ours</TableHead>
                <TableHead className="text-right">Paper</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {TRANSFER.map((r) => (
                <TableRow key={r.label}>
                  <TableCell>{r.label}</TableCell>
                  <TableCell className="readout text-right">{r.ours}</TableCell>
                  <TableCell className="readout text-muted-foreground text-right">
                    {r.paper}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <p className="text-muted-foreground text-xs">
            The <span className="readout">0.6007</span> is on clean ground-truth tracks; our
            pass-only truth chain gives <span className="readout">0.5611</span>. Naming is
            ahead of the paper on the sender and behind it on the receiver, and the two
            possession-share errors are the same size in opposite directions. This is one
            match — read it as evidence that the transfer works at all, not as an improvement
            on the paper.
          </p>
        </div>

        <div className="grid gap-2 sm:grid-cols-3">
          {TRANSFER_ONLY.map((t) => (
            <div key={t.label} className="bg-muted rounded-[var(--radius-md)] p-3">
              <div className="readout text-lg leading-6 font-medium">{t.n}</div>
              <div className="text-muted-foreground mt-1 text-[11px] leading-snug">
                {t.label}
                <span className="block">{t.sub}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="grid min-w-0 gap-2">
          <h3 className="text-sm font-medium">
            Sensitivity to positional error — this is what gates everything
          </h3>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Positional error added</TableHead>
                <TableHead className="text-right">Event detection F1</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {SENSITIVITY.map(([err, f1]) => (
                <TableRow key={err}>
                  <TableCell className="readout">{err}</TableCell>
                  <TableCell className="readout text-right">{f1}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <p className="text-muted-foreground text-xs">
            Half a metre of positional error costs <span className="readout">0.04</span> F1; a
            full metre costs <span className="readout">0.14</span>. Every event figure above
            therefore inherits whatever positional error the pipeline actually has — which is
            the quantity this repo has never measured end to end over a half.
          </p>
        </div>

        <div className="touchline grid gap-2 pt-4">
          <h3 className="text-sm font-medium">Why there is no player-level output</h3>
          <p className="text-[13px] leading-relaxed">
            Charge a perfect model with nothing but our identity error and sender accuracy
            falls from <span className="readout">100%</span> to{" "}
            <b className="readout text-bad">9.1%</b>, against a chance floor of{" "}
            <span className="readout">4.5%</span>. Pass-network degree error comes out at{" "}
            <b className="readout text-bad">8.09</b>, larger than the{" "}
            <span className="readout">7.36</span> of signal it is supposed to be measuring.
          </p>
          <p className="text-muted-foreground text-xs">
            The model is not the limit here; naming the players is. A per-player passing
            number would carry more error than content, so the product does not offer one —
            it is withheld deliberately, not missing.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

// ------------------------------------------------------- not established

/** The four things the block above does NOT establish. Same card, same type
 *  size, same weight as the evidence — a page of evidence that sets its own
 *  gaps in small print at the bottom is worse than no page. */
const NOT_ESTABLISHED: [string, string][] = [
  [
    "Phase 1 has never been run on real tracker output over a full half.",
    "Every transfer figure above was measured on tracks, not on this detector and this tracker running a half end to end. That run has not happened, so what the product does is unmeasured — not estimated, unmeasured.",
  ],
  [
    "The best end-to-end positional figure, 1.33 m, is 24 frames.",
    "Per-frame and unfiltered, on the smallest sample anywhere on this page. It is an indication that the chain connects. It is not a positional accuracy for the pipeline and must not be quoted as one.",
  ],
  [
    "The smoothing window was fitted on the test match, not held out.",
    "It was chosen against the same match it is reported on. Every figure that depends on it is therefore optimistic by an amount nobody here can state.",
  ],
  [
    "No second match has an event feed.",
    "Nothing above is validated across matches. One match, one camera, one venue: a figure that has not survived a second match has not yet been shown to be a figure.",
  ],
];

function NotEstablished() {
  return (
    <Card className="border-t-warn gap-4 border-t-3">
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
          Not established
          <Badge variant="outline" className="text-warn">
            {NOT_ESTABLISHED.length}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4 sm:grid-cols-2">
        <p className="text-[13px] leading-relaxed sm:col-span-2">
          These carry the same weight as the numbers above them, and they are listed here
          rather than in a footnote because they are the reason those numbers are not a
          product claim. Nothing on this page should be read as if any of the four were
          settled.
        </p>
        {NOT_ESTABLISHED.map(([claim, why]) => (
          <div key={claim} className="grid gap-1">
            <h3 className="text-warn text-sm leading-snug font-medium">{claim}</h3>
            <p className="text-muted-foreground text-[13px] leading-snug">{why}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------- verdict

function Verdict({ report, rel }: { report: ValidationReport; rel: string }) {
  const head = report.headline;
  const keys = Object.keys(head);
  const synthetic = isSynthetic(report);

  const notEvaluated = keys.filter((k) => head[k].verdict === "NOT EVALUATED");
  const failed = keys.filter((k) =>
    ["FAIL", "MISSED", "MARGINAL"].includes(head[k].verdict),
  );
  const met = keys.filter((k) => ["PASS", "MET"].includes(head[k].verdict));
  // Nothing from a synthetic run is allowed into "safe to act on" no matter how
  // good the verdict looks. This page is the one that decides what may be said.
  const safe = synthetic ? [] : met;
  const uncertifying = synthetic ? met : [];

  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 text-base">
          Verdict
          <Badge variant="outline" className="text-warn">
            gate {report.gate}, on a test that cannot certify
          </Badge>
        </CardTitle>
      </CardHeader>

      <CardContent className="grid gap-4">
        <Alert>
          <AlertTitle>
            {synthetic
              ? "This run compared two estimators over identical positions."
              : `${failed.length} of ${keys.length} metrics missed their target.`}
          </AlertTitle>
          <AlertDescription className="text-muted-foreground">
            {synthetic && (
              <>
                The measured side was built by perturbing the ground truth, so both sides
                are the same positions. That can show the maths is sound. It cannot show
                the product is accurate — <b>the tracker has never run on real side-on
                footage</b> — and no figure from it belongs in front of a club.{" "}
              </>
            )}
            {notEvaluated.length > 0 && (
              <>
                <b>
                  {notEvaluated.length} metric{notEvaluated.length === 1 ? "" : "s"}{" "}
                  reported NOT EVALUATED
                </b>{" "}
                — the harness could not perturb{" "}
                {notEvaluated.length === 1 ? "it" : "them"}, so the test could not have
                failed and a pass would have meant nothing.
              </>
            )}
          </AlertDescription>
        </Alert>

        <div className="grid items-start gap-3 md:grid-cols-3">
          <Bucket
            tone="border-t-ok"
            title="Safe to act on"
            count={safe.length}
            caption="Passed a held-out test against an independent reference."
          >
            {safe.length ? (
              safe.map((k) => <Item key={k} title={METRIC_NAME[k] ?? k} sub={figure(head[k])} />)
            ) : (
              <li className="text-muted-foreground text-xs">
                Nothing. There is no independent reference in this repo to pass a test
                against.
              </li>
            )}
            <Item
              title="Ranks and shares within one match"
              sub="Not separately validated, but a ratio between two players measured by the same instrument cancels most of a systematic bias. This is the strongest thing the match report can say about a failing metric."
            />
          </Bucket>

          <Bucket
            tone="border-t-bad"
            title="Do not quote"
            count={failed.length + uncertifying.length}
            caption="Missed its target, or met one that could not certify anything."
          >
            {failed.map((k) => (
              <Item
                key={k}
                title={METRIC_NAME[k] ?? k}
                sub={`${head[k].verdict} · ${figure(head[k])}`}
              />
            ))}
            {uncertifying.map((k) => (
              <Item
                key={k}
                title={METRIC_NAME[k] ?? k}
                sub={`${head[k].verdict} · ${figure(head[k])} — ${whyNotCertifying(head[k], report)}`}
              />
            ))}
          </Bucket>

          <Bucket
            tone="border-t-muted-foreground"
            title="Never measured"
            count={notEvaluated.length + NEVER_MEASURED.length}
            caption="Not failing. Not passing. No evidence exists either way."
          >
            {notEvaluated.map((k) => (
              <Item
                key={k}
                title={METRIC_NAME[k] ?? k}
                sub={`NOT EVALUATED · the harness could not perturb this, so the test could not have failed. n = ${head[k].n} is the pairing count, not evidence.`}
              />
            ))}
            {NEVER_MEASURED.map(([t, s]) => (
              <Item key={t} title={t} sub={s} />
            ))}
          </Bucket>
        </div>

        <div className="text-muted-foreground flex flex-wrap gap-x-5 gap-y-1 text-xs">
          <span>
            report <b className="text-foreground">{rel}</b>
          </span>
          <span>
            n{" "}
            <b className={report.n_player_matches >= report.n_required ? "text-foreground" : "text-bad"}>
              {report.n_player_matches} / {report.n_required} required
            </b>
          </span>
          <span>
            held out{" "}
            <b className="text-foreground">{report.held_out_matches.length || "none"}</b>
          </span>
          <span>
            bandwidth{" "}
            <b className="text-foreground">{String(report.config.distance_window_s)} s</b>
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

function whyNotCertifying(r: MetricResult, report: ValidationReport): string {
  const parts = [
    "measured against a perturbed copy of its own input, so this is agreement between two estimators rather than a measurement of the pipeline",
  ];
  if (r.n < report.n_required)
    parts.push(`n = ${r.n}, far below the n ≥ ${report.n_required} required before publishing`);
  return parts.join("; ") + ".";
}

function figure(r: MetricResult): string {
  const mae =
    r.mae_pct == null
      ? "–"
      : r.mae_pct >= 1000
        ? `${Math.round(r.mae_pct).toLocaleString("en-US")}%`
        : `${r.mae_pct.toFixed(1)}%`;
  const unit = r.metric === "id_switches" ? " per player" : "%";
  return `MAE ${mae} against a ≤${r.target}${unit} target · n = ${r.n} · ${
    r.gates ? "gating" : "report-only"
  }`;
}

function Bucket({
  tone,
  title,
  count,
  caption,
  children,
}: {
  tone: string;
  title: string;
  count: number;
  caption: string;
  children: React.ReactNode;
}) {
  return (
    <div className={`bg-muted rounded-[var(--radius-md)] border-t-3 p-3 ${tone}`}>
      <h3 className="flex items-center gap-2 text-sm font-medium">
        {title}
        <Badge variant="secondary">{count}</Badge>
      </h3>
      <p className="text-muted-foreground mt-1 mb-3 text-[11px] leading-relaxed">{caption}</p>
      <ul className="grid gap-2.5">{children}</ul>
    </div>
  );
}

function Item({ title, sub }: { title: string; sub: string }) {
  return (
    <li className="text-[13px] leading-tight">
      <b className="font-medium">{title}</b>
      <span className="text-muted-foreground mt-1 block text-[11px] leading-snug">{sub}</span>
    </li>
  );
}

// ----------------------------------------------------------------- detail

function Detail({ report, rel }: { report: ValidationReport; rel: string }) {
  const gating = Object.values(report.headline).filter((r) => r.gates);
  const reportOnly = Object.values(report.headline).filter((r) => !r.gates);

  return (
    <Card className="min-w-0 gap-4">
      <CardHeader>
        <CardTitle className="text-sm">The measurement behind that verdict</CardTitle>
      </CardHeader>
      <CardContent className="grid min-w-0 gap-5">
        <div className="grid gap-3 text-sm md:grid-cols-2">
          <div>
            <div className="text-muted-foreground text-xs">Reference</div>
            <p className="mt-1 text-[13px] leading-snug">{report.source}</p>
            <p className="text-muted-foreground mt-1 text-xs">
              Its own error is stated at {report.reference_error_pct ?? "–"}%. Every figure
              below is agreement between two imperfect instruments, not error against
              truth — the limits of agreement column is the honest read.
            </p>
          </div>
          <div className="min-w-0">
            <div className="text-muted-foreground text-xs">
              Configuration that produced it — a figure here describes THIS configuration
              and no other
            </div>
            <dl className="mt-1 grid grid-cols-2 gap-x-4 text-xs">
              {Object.entries(report.config).map(([k, v]) => (
                <div key={k} className="flex justify-between gap-2 border-b py-1">
                  <dt className="text-muted-foreground truncate">{k}</dt>
                  <dd className="font-mono">{String(v)}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>

        <MetricTable title="Gating metrics (held-out)" rows={gating} />
        <MetricTable title="Report-only metrics (held-out)" rows={reportOnly} />

        <p className="text-muted-foreground text-xs">
          MAE is the headline; TEE, CCC, bias and the limits of agreement are what a
          club&apos;s performance staff actually ask for. A small bias with wide limits
          means individual players are badly wrong in both directions and it cancels in the
          mean. A <b>gating</b> metric blocks the claim; the rest are report-only.
        </p>

        <Accordion type="multiple" className="min-w-0">
          <AccordionItem value="caveats">
            <AccordionTrigger>
              Caveats ({report.caveats.length}) — read before quoting anything above
            </AccordionTrigger>
            <AccordionContent>
              <ul className="text-muted-foreground grid gap-2 text-xs">
                {report.caveats.map((c, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-muted-foreground/60">—</span>
                    <span>{stripEmphasis(c)}</span>
                  </li>
                ))}
              </ul>
            </AccordionContent>
          </AccordionItem>

          {report.fitted_train_split && (
            <AccordionItem value="fitted">
              <AccordionTrigger>Fitted train split — not a measurement</AccordionTrigger>
              <AccordionContent className="grid gap-3">
                <p className="text-muted-foreground text-xs">
                  These matches tuned the estimator ({report.train_matches.join(", ")}).
                  Quoting them as accuracy is quoting the model&apos;s memory of its own
                  training data. They are here only so the gap against the held-out figures
                  is visible.
                </p>
                <MetricTable
                  title=""
                  rows={Object.values(report.fitted_train_split)}
                />
              </AccordionContent>
            </AccordionItem>
          )}

          <AccordionItem value="provenance">
            <AccordionTrigger>Where this page gets its numbers</AccordionTrigger>
            <AccordionContent>
              <ul className="text-muted-foreground grid gap-2 text-xs">
                <li>
                  Read from <code className="bg-muted rounded px-1 py-0.5">{rel}</code> on
                  the server at request time, not fetched by the browser and not frozen
                  into a build.
                </li>
                <li>
                  Held-out matches: {report.held_out_matches.join(", ") || "none"}.
                </li>
                <li>
                  Overall status recorded by the harness: {report.status}.
                </li>
                <li>
                  <code className="bg-muted rounded px-1 py-0.5">model/validation/retired/</code>{" "}
                  is never read. Those reports were found to be measuring something other
                  than what their filename implied and are kept only as evidence.
                </li>
              </ul>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
}

const VERDICT_CLASS: Record<string, string> = {
  PASS: "text-ok",
  MET: "text-ok",
  MARGINAL: "text-warn",
  FAIL: "text-bad",
  MISSED: "text-bad",
};

function MetricTable({ title, rows }: { title: string; rows: MetricResult[] }) {
  const num = (v: number | null, d: number) => (v == null ? "–" : v.toFixed(d));
  return (
    <div className="grid min-w-0 gap-2">
      {title && <h3 className="text-sm font-medium">{title}</h3>}
      {/* Every wide table scrolls inside its own box: a page that pushes the
          viewport sideways hides the column you came to read. */}
      <div className="w-full overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Metric</TableHead>
              <TableHead className="text-right">n</TableHead>
              <TableHead className="text-right">MAE %</TableHead>
              <TableHead className="text-right">Target</TableHead>
              <TableHead className="text-right">TEE %</TableHead>
              <TableHead className="text-right">CCC</TableHead>
              <TableHead className="text-right">Bias %</TableHead>
              <TableHead className="text-right">95% limits of agreement</TableHead>
              <TableHead className="text-right">Verdict</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => (
              <TableRow key={r.metric}>
                <TableCell className="whitespace-nowrap">
                  {METRIC_NAME[r.metric] ?? r.metric}
                  {r.skipped_zero_gt > 0 && (
                    <span className="text-muted-foreground ml-1 text-[11px]">
                      ({r.skipped_zero_gt} skipped, zero in reference)
                    </span>
                  )}
                </TableCell>
                <TableCell className="text-right tabular-nums">{r.n}</TableCell>
                <TableCell className="text-right tabular-nums">{num(r.mae_pct, 2)}</TableCell>
                <TableCell className="text-muted-foreground text-right tabular-nums">
                  ≤{num(r.target, 0)}
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {num(r.tee_pct, 2)}
                  {r.tee_rating !== "n/a" && (
                    <span className="text-muted-foreground ml-1 text-[11px]">
                      {r.tee_rating}
                    </span>
                  )}
                </TableCell>
                <TableCell className="text-right tabular-nums">{num(r.ccc, 3)}</TableCell>
                <TableCell className="text-right tabular-nums">{num(r.bias_pct, 2)}</TableCell>
                <TableCell className="text-muted-foreground text-right whitespace-nowrap tabular-nums">
                  {num(r.loa_lower, 1)} … {num(r.loa_upper, 1)}
                </TableCell>
                <TableCell
                  className={`text-right whitespace-nowrap ${VERDICT_CLASS[r.verdict] ?? "text-muted-foreground"}`}
                >
                  {r.verdict}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

/** The caveats are markdown with **bold** in them. One regex beats a parser. */
function stripEmphasis(s: string): React.ReactNode {
  return s.split(/\*\*(.+?)\*\*/g).map((part, i) =>
    i % 2 ? (
      <b key={i} className="text-foreground">
        {part}
      </b>
    ) : (
      part
    ),
  );
}

function NoReport() {
  return (
    <Alert variant="destructive">
      <AlertTitle>No validation report on disk.</AlertTitle>
      <AlertDescription>
        Nothing has been compared against ground truth. Accuracy is unknown — not
        &ldquo;probably fine&rdquo;, unknown. Expected it at{" "}
        <code className="bg-muted rounded px-1 py-0.5">{REPORT_REL}</code> relative to the
        repository root.
      </AlertDescription>
    </Alert>
  );
}
