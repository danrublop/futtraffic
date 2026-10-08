/** EVERY BLOCK, ON FAKE DATA, IN ONE NOTE.
 *
 *  A smoke test you look at rather than assert on. The unit tests pin each
 *  renderer's markup; this pins the thing they cannot - that all of them
 *  together still make a document a coach would accept, on paper, at A4.
 *
 *  FAKE ON PURPOSE. Nothing here touches the store, so it runs on an empty
 *  install and it cannot go stale when the demo data changes. The names are
 *  invented; any resemblance to a real squad is an accident of the alphabet.
 *
 *      npx vite-node --config vitest.config.mts scripts/block-gallery.mts
 *
 *  Writes to `Test/Block gallery.html` in the note library and drops a PDF
 *  beside it. Delete the folder when you are done; nothing else refers to it.
 */
import {
  writeNote,
  writePdf,
  h,
  p,
  ul,
  quote,
  hr,
  page,
  pageSetup,
  // No pageBreak: a `page()` already breaks before itself, and putting one
  // between two of them prints a blank sheet. See its docstring.
  pitch,
  chart,
  goalMouth,
  pizza,
  radar,
  freeze,
  passes,
  row,
  table,
  typable,
  face,
  type Block,
} from "./note-api.mts";
import { freezeFrame, passSet, percentileSlices } from "./demo-football.mts";

const OUT_NOTE = "Test/Block gallery.html";
const OUT_PDF = process.env.GALLERY_PDF ?? "/tmp/block-gallery.pdf";

// ---------------------------------------------------------------- the board
// A corner, drawn the way a coach would: the ball, the near-post run, the
// blocker, and the two who stay out for the second ball.
const board = pitch({
  view: "full",
  magnets: [
    { x: 99, y: 97, label: "", side: "ball" },
    { x: 88, y: 46, label: "9", side: "home" },
    { x: 84, y: 55, label: "5", side: "home" },
    { x: 92, y: 38, label: "4", side: "home" },
    { x: 70, y: 50, label: "8", side: "home" },
    { x: 62, y: 30, label: "7", side: "home" },
    { x: 90, y: 50, label: "6", side: "away" },
    { x: 86, y: 42, label: "3", side: "away" },
    { x: 95, y: 48, label: "1", side: "away" },
    { x: 66, y: 44, label: "10", side: "away" },
  ],
  shapes: [
    { kind: "pass", pts: [99, 95, 89, 47], color: "ink" },
    { kind: "run", pts: [70, 50, 86, 44], color: "ink" },
    { kind: "dribble", pts: [62, 30, 74, 26], color: "ink" },
    { kind: "zone", pts: [80, 34, 96, 60], color: "ink" },
    { kind: "text", pts: [58, 22], color: "ink", label: "second ball" },
  ],
  caption: "Corner from the right: 8 attacks the near post, 7 holds for the second ball",
});

// --------------------------------------------------------------- the charts
const shotsByZone = chart({
  title: "Shots by zone",
  render: "column",
  theme: "report",
  rows: [
    { label: "Six-yard", values: [3] },
    { label: "Penalty area", values: [11] },
    { label: "Outside", values: [8] },
  ],
  dimLabel: "Zone",
  label: "Shots",
});

const xgRace = chart({
  title: "Cumulative xG",
  render: "line",
  ordered: true,
  unit: "",
  dp: 2,
  series: ["Home", "Away"],
  rows: [
    { label: "0", values: [0, 0] },
    { label: "15", values: [0.12, 0.31] },
    { label: "30", values: [0.44, 0.38] },
    { label: "45", values: [0.61, 0.52] },
    { label: "60", values: [1.05, 0.55] },
    { label: "75", values: [1.28, 0.9] },
    { label: "90", values: [1.71, 0.94] },
  ],
});

const outcome = chart({
  title: "How the shots ended",
  render: "doughnut",
  theme: "report",
  rows: [
    { label: "On target", values: [9] },
    { label: "Off target", values: [8] },
    { label: "Blocked", values: [5] },
  ],
});

const minutesVsDistance = chart({
  title: "Minutes against distance",
  render: "scatter",
  series: ["Minutes", "Distance"],
  dp: 1,
  rows: [
    { label: "Aliyev", values: [90, 11.4] },
    { label: "Brandt", values: [90, 12.1] },
    { label: "Costa", values: [74, 9.8] },
    { label: "Duarte", values: [61, 8.2] },
    { label: "Eriksen", values: [45, 5.9] },
    { label: "Faber", values: [30, 3.6] },
  ],
});

const pressures = chart({
  title: "Pressures and recoveries by third",
  render: "combo",
  series: ["Pressures", "Recoveries"],
  rows: [
    { label: "Defensive", values: [22, 14] },
    { label: "Middle", values: [41, 19] },
    { label: "Attacking", values: [28, 7] },
  ],
});

// ------------------------------------------------------- football natives
const penalties = goalMouth({
  title: "Penalties faced, this season",
  who: "M. Novak",
  shots: [
    { x: 14, y: 28, scored: true, label: "1" },
    { x: 86, y: 33, scored: true, label: "2" },
    { x: 22, y: 24, scored: false, label: "3" },
    { x: 78, y: 71, scored: true, label: "4" },
    { x: 50, y: 92, scored: false, label: "5" },
    { x: 9, y: 60, scored: false, label: "6" },
  ],
});

const percentile = pizza({
  title: "T. Brandt against the league",
  who: "T. Brandt",
  population: 214,
  tier: "prov",
  slices: [
    { label: "Distance", pct: 88, text: "12.1 km", group: "Volume" },
    { label: "Minutes", pct: 94, text: "2610", group: "Volume" },
    { label: "Top speed", pct: 71, text: "33.4 km/h", group: "Intensity" },
    { label: "Sprints", pct: 63, text: "41", group: "Intensity" },
    { label: "Pressures", pct: 34, text: "112", group: "Duels" },
    { label: "Duels won", pct: 22, text: "48%", group: "Duels" },
  ],
});

// --------------------------------------------------------------- the tables
const squad = table({
  head: ["Player", "Min", "Dist km", "Sprints", "Share"],
  rows: [
    ["Aliyev", 90, "11.4", 38, "9%"],
    ["Brandt", 90, "12.1", 41, "10%"],
    ["Costa", 74, "9.8", 29, "8%"],
    ["Duarte", 61, "8.2", 22, "7%"],
    ["Eriksen", 45, "5.9", 15, "5%"],
  ],
  variant: "report",
  bars: [null, null, null, null, 0.1],
  tones: [
    ["", "num", "num", "num", "num"],
    ["", "num", "num", "num", "num"],
    ["", "num", "num", "num", "num"],
    ["", "num", "num", "num", "num"],
    ["", "num", "num", "num", "num"],
  ],
});

const toFill = typable(
  ["Player", "Rating", "Note"],
  [
    ["Aliyev", "", ""],
    ["Brandt", "", ""],
    ["Costa", "", ""],
  ],
);

// ------------------------------------------------------- the generated half
// Everything below carries synthetic:true and prints "generated, not measured"
// on the figure. The store holds no frame slice and no pass, so these figures
// have nothing real to draw yet - see demo-football.mts.
const full = freezeFrame();
const partial = freezeFrame({ visible: 13 });
const pset = passSet();
const player = {
  who: "T. Brandt",
  population: 214,
  tier: "prov",
  slices: percentileSlices(),
};

// ----------------------------------------------------------------- the note
const doc: Block[] = [
  pageSetup("landscape"),
  page([
    h(1, "Block gallery"),
    p(
      "Every block the notepad can draw, on invented numbers. If a figure below is " +
        "blank, squashed, or has a label sitting on top of a value, that is the bug.",
    ),
    hr(),
    // The board gets two thirds and the words one — the proportion the report
    // this is modelled on uses, and the reason `row` takes weights at all.
    row([board, ul([
      "The board is inline SVG, so it survives being mailed as a file.",
      "Nothing on this page touched the match store.",
      "Delete the Test folder when you are done with it.",
    ])], [2, 1]),
  ]),
  page([h(2, "Charts"), row([shotsByZone, outcome])]),
  page([h(2, "Two measures"), row([xgRace, minutesVsDistance])]),
  page([
    h(2, "Drawn on the pitch, and against a population"),
    row([penalties, percentile]),
    row([pressures, [
      p(
        "Combo draws both series against one maximum. A second value axis is the " +
          "most misread chart there is: where the series cross is decided by " +
          "whoever set the ranges.",
      ),
      p("A percentile of a provisional number is still provisional."),
    ]], [2, 1]),
  ]),
  page([
    h(2, "One moment, three ways"),
    p(
      "All three read the same frame slice. The line under each says how many of " +
        "the squad were visible, because a shape drawn from part of a side is not " +
        "that side's shape.",
    ),
    // THREE UP, not two rows of two. A pitch is 1.5:1 against a 1.41:1 sheet,
    // so one alone can never fill a page and two rows of them overflow it —
    // measured at 33mm over, by the check in writePdf. Three across is the
    // small-multiple shape: identical scale, identical size, read as one.
    row([
      freeze({ ...full, render: "freeze", title: "Freeze frame" }),
      freeze({ ...full, render: "hull", title: "Team shape" }),
      freeze({ ...full, render: "control", title: "Pitch control" }),
    ]),
    p(
      "Pitch control is a nearest-side grid, not a Voronoi — at print size the " +
        "picture is the same and the claim is smaller. The hull beside it is the " +
        "polygon stats.py computes the area of and has never drawn.",
    ),
  ]),
  page([
    h(2, "And the same moment with the camera on the ball"),
    p(
      "Thirteen of twenty-two. This is what every figure above looks like on " +
        "ball-following footage, and it is why the count is printed rather than assumed.",
    ),
    row([
      freeze({ ...partial, render: "hull", title: "Team shape, 13 of 22 seen" }),
      freeze({ ...partial, render: "control", title: "Pitch control, 13 of 22 seen" }),
    ]),
  ]),
  page([
    h(2, "Passing"),
    row([
      passes({ ...pset, render: "network", title: "Pass network" }),
      passes({ ...pset, render: "progressive", title: "Every pass, and its outcome" }),
      passes({ ...pset, render: "sonar", title: "Pass sonar" }),
    ]),
    p(
      "Undirected edges on the network: a pairing is one relationship, and drawing " +
        "both directions makes a two-way partnership look like two weak ones. The " +
        "sonar is a rose per player at his mean passing position.",
    ),
  ]),
  page([
    h(2, "The same player, two ways"),
    row([
      pizza({ ...player, title: "As a pizza — for a coach" }),
      radar({ ...player, title: "As a radar — for an analyst" }),
    ]),
    quote(
      "A radar's enclosed area reads as a quantity and is not one: reorder the " +
        "axes and the same player encloses a different area. The pizza joins nothing.",
    ),
  ]),
  page([
    h(2, "Tables and faces"),
    squad,
    p("And one a coach fills in himself:"),
    toFill,
    face({ name: "7. T. Brandt", sub: "24 years, 181 cm, right" }),
  ]),
];

const bytes = await writeNote(OUT_NOTE, doc);
console.log(`${(bytes / 1024).toFixed(1)} KB -> ${OUT_NOTE}`);
const pdf = await writePdf(OUT_PDF, doc, { title: "Block gallery", sub: "invented data" });
console.log(`${(pdf / 1024).toFixed(1)} KB -> ${OUT_PDF}`);
