/** THE PICK-LISTS FOR A PLAYER PROFILE: nations, positions, feet.
 *
 *  WHY THESE ARE TYPED OUT AND NOT DERIVED FROM THE DATA. A dropdown built
 *  from the bios already loaded can only offer a country somebody in the club
 *  already has - which is exactly wrong for the club this product is for,
 *  where the next signing is the first player from that country. The list has
 *  to exist before the player does.
 *
 *  WHY FIFA CODES AND NOT ISO. The feed writes NED, POR, ENG - Netherlands,
 *  Portugal and England - where ISO 3166 writes NLD, PRT and GBR. Half these
 *  bios came from that feed, so a second alphabet here would mean the same
 *  nation reading two ways depending on who typed it. Codes below are the FIFA
 *  association codes, and the four home nations are separate associations, as
 *  they are in the feed.
 *
 *  Related: jersey-numbers-come-from-the-operator (internal note) - the entered
 *  half of a profile is typed by a person, and this is that person's keyboard.
 */

/** "FIFA ISO Name", one per line. A table, not 200 object literals, because
 *  the thing a reader checks here is a column of codes against a column of
 *  names and an object per row hides both behind punctuation.
 *
 *  THE SECOND COLUMN IS THE FLAG'S KEY, and it is a different alphabet from
 *  the first on purpose: ISO 3166-1 alpha-2, lowercased, which is what
 *  flag-icons files its SVGs under. The four home nations are not ISO
 *  countries at all - they are gb-eng, gb-sct, gb-wls and gb-nir, which that
 *  set carries as subdivisions. Keeping both columns is what lets the profile
 *  store the feed's code and still draw a flag
 *  (detect-on-the-raw-panorama-not-the-vendor-remap (internal note) is not
 *  this, but the same lesson: one id per consumer, mapped, never guessed). */
const NATION_TABLE = `
AFG AF Afghanistan
ALB AL Albania
ALG DZ Algeria
ASA AS American Samoa
AND AD Andorra
ANG AO Angola
AIA AI Anguilla
ATG AG Antigua and Barbuda
ARG AR Argentina
ARM AM Armenia
ARU AW Aruba
AUS AU Australia
AUT AT Austria
AZE AZ Azerbaijan
BAH BS Bahamas
BHR BH Bahrain
BAN BD Bangladesh
BRB BB Barbados
BLR BY Belarus
BEL BE Belgium
BLZ BZ Belize
BEN BJ Benin
BER BM Bermuda
BHU BT Bhutan
BOL BO Bolivia
BIH BA Bosnia and Herzegovina
BOT BW Botswana
BRA BR Brazil
VGB VG British Virgin Islands
BRU BN Brunei
BUL BG Bulgaria
BFA BF Burkina Faso
BDI BI Burundi
CAM KH Cambodia
CMR CM Cameroon
CAN CA Canada
CPV CV Cape Verde
CAY KY Cayman Islands
CTA CF Central African Republic
CHA TD Chad
CHI CL Chile
CHN CN China PR
TPE TW Chinese Taipei
COL CO Colombia
COM KM Comoros
CGO CG Congo
COK CK Cook Islands
CRC CR Costa Rica
CIV CI Cote d'Ivoire
CRO HR Croatia
CUB CU Cuba
CUW CW Curacao
CYP CY Cyprus
CZE CZ Czechia
COD CD DR Congo
DEN DK Denmark
DJI DJ Djibouti
DMA DM Dominica
DOM DO Dominican Republic
ECU EC Ecuador
EGY EG Egypt
SLV SV El Salvador
ENG gb-eng England
EQG GQ Equatorial Guinea
ERI ER Eritrea
EST EE Estonia
SWZ SZ Eswatini
ETH ET Ethiopia
FRO FO Faroe Islands
FIJ FJ Fiji
FIN FI Finland
FRA FR France
GUF GF French Guiana
GUM GU Guam
GAB GA Gabon
GAM GM Gambia
GEO GE Georgia
GER DE Germany
GHA GH Ghana
GIB GI Gibraltar
GRE GR Greece
GRN GD Grenada
GUA GT Guatemala
GUI GN Guinea
GNB GW Guinea-Bissau
GUY GY Guyana
HAI HT Haiti
HON HN Honduras
HKG HK Hong Kong
HUN HU Hungary
ISL IS Iceland
IND IN India
IDN ID Indonesia
IRN IR Iran
IRQ IQ Iraq
IRL IE Ireland
ISR IL Israel
ITA IT Italy
JAM JM Jamaica
JPN JP Japan
JOR JO Jordan
KAZ KZ Kazakhstan
KEN KE Kenya
KIR KI Kiribati
KOS XK Kosovo
KUW KW Kuwait
KGZ KG Kyrgyzstan
LAO LA Laos
LVA LV Latvia
LBN LB Lebanon
LES LS Lesotho
LBR LR Liberia
LBY LY Libya
LIE LI Liechtenstein
LTU LT Lithuania
LUX LU Luxembourg
MAC MO Macau
MAD MG Madagascar
MWI MW Malawi
MAS MY Malaysia
MDV MV Maldives
MLI ML Mali
MLT MT Malta
MTN MR Mauritania
MRI MU Mauritius
MEX MX Mexico
MDA MD Moldova
MNG MN Mongolia
MNE ME Montenegro
MSR MS Montserrat
MAR MA Morocco
MOZ MZ Mozambique
MYA MM Myanmar
NAM NA Namibia
NEP NP Nepal
NED NL Netherlands
NCL NC New Caledonia
NZL NZ New Zealand
NCA NI Nicaragua
NIG NE Niger
NGA NG Nigeria
PRK KP North Korea
MKD MK North Macedonia
NIR gb-nir Northern Ireland
NOR NO Norway
OMA OM Oman
PAK PK Pakistan
PLE PS Palestine
PAN PA Panama
PNG PG Papua New Guinea
PAR PY Paraguay
PER PE Peru
PHI PH Philippines
POL PL Poland
POR PT Portugal
PUR PR Puerto Rico
QAT QA Qatar
ROU RO Romania
RUS RU Russia
RWA RW Rwanda
SAM WS Samoa
SMR SM San Marino
STP ST Sao Tome and Principe
KSA SA Saudi Arabia
SCO gb-sct Scotland
SEN SN Senegal
SRB RS Serbia
SEY SC Seychelles
SLE SL Sierra Leone
SGP SG Singapore
SVK SK Slovakia
SVN SI Slovenia
SOL SB Solomon Islands
SOM SO Somalia
RSA ZA South Africa
KOR KR South Korea
SSD SS South Sudan
ESP ES Spain
SRI LK Sri Lanka
SKN KN St Kitts and Nevis
LCA LC St Lucia
VIN VC St Vincent and the Grenadines
SDN SD Sudan
SUR SR Suriname
SWE SE Sweden
SUI CH Switzerland
SYR SY Syria
TAH PF Tahiti
TJK TJ Tajikistan
TAN TZ Tanzania
THA TH Thailand
TLS TL Timor-Leste
TOG TG Togo
TGA TO Tonga
TRI TT Trinidad and Tobago
TUN TN Tunisia
TUR TR Turkiye
TKM TM Turkmenistan
TCA TC Turks and Caicos Islands
UGA UG Uganda
UKR UA Ukraine
UAE AE United Arab Emirates
USA US United States
URU UY Uruguay
VIR VI US Virgin Islands
UZB UZ Uzbekistan
VAN VU Vanuatu
VEN VE Venezuela
VIE VN Vietnam
WAL gb-wls Wales
YEM YE Yemen
ZAM ZM Zambia
ZIM ZW Zimbabwe
`;

export interface Nation {
  /** The FIFA association code the feed writes, e.g. "NED". */
  code: string;
  /** The flag-icons key: ISO alpha-2 lowercased, or "gb-eng" and friends. */
  iso: string;
  name: string;
}

/** Every nation, in the table's own order (alphabetical by name). */
export const NATIONS: Nation[] = NATION_TABLE.trim()
  .split("\n")
  .map((line) => {
    const [code, iso, ...rest] = line.split(" ");
    return { code, iso: iso.toLowerCase(), name: rest.join(" ") };
  });

const BY_NAME = new Map(NATIONS.map((n) => [n.name.toLowerCase(), n]));

/** The code for a country name as the feed writes it, or null.
 *
 *  Matched on the NAME because that is the field the panel shows and the one
 *  an operator picks; the code is the derived half and is filled in from here
 *  so a hand-entered player's flag keys the same way a feed player's does. */
export function nationCode(name: string | null | undefined): string | null {
  return BY_NAME.get((name ?? "").trim().toLowerCase())?.code ?? null;
}

/** The flag key for a country name, or null for a country not in the table -
 *  and null means DRAW NOTHING. A flag lookup that falls back to something
 *  near enough prints another country's flag against a player's name, which
 *  is worse than a blank (an internal note is emphatic about the home nations:
 *  England is not GB). */
export function nationFlag(name: string | null | undefined): string | null {
  return BY_NAME.get((name ?? "").trim().toLowerCase())?.iso ?? null;
}

/** The positions a player can be filed under, in the order a team sheet reads
 *  them - back to front, not alphabetical, because that is how a coach scans
 *  for the gap in a squad. Short codes are what the panel's Other positions
 *  row already prints ("RB, CM, AM"). */
export const POSITIONS: { short: string; label: string }[] = [
  { short: "GK", label: "Goalkeeper" },
  { short: "RB", label: "Right Back" },
  { short: "CB", label: "Centre Back" },
  { short: "LB", label: "Left Back" },
  { short: "RWB", label: "Right Wing Back" },
  { short: "LWB", label: "Left Wing Back" },
  { short: "DM", label: "Defensive Midfielder" },
  { short: "CM", label: "Centre Midfielder" },
  { short: "AM", label: "Attacking Midfielder" },
  { short: "RM", label: "Right Midfielder" },
  { short: "LM", label: "Left Midfielder" },
  { short: "RW", label: "Right Winger" },
  { short: "LW", label: "Left Winger" },
  { short: "SS", label: "Second Striker" },
  { short: "ST", label: "Striker" },
];

const BY_SHORT = new Map(POSITIONS.map((p) => [p.short.toLowerCase(), p]));
const BY_LABEL = new Map(POSITIONS.map((p) => [p.label.toLowerCase(), p]));

/** TWO ALPHABETS, AND THE TRANSLATION BETWEEN THEM.
 *
 *  A position is written two ways in this product and both are load-bearing:
 *  the BIO carries the feed's long label ("Centre Back"), which is what the
 *  profile prints, and the ROSTER carries the short code ("CB"), which is what
 *  `group()` in players-section.tsx parses to decide the badge, the sort rail
 *  and the squad filters.
 *
 *  Writing the wrong one is silent and wrong: group() falls through to MID for
 *  anything it does not recognise, so storing "Centre Back" on the roster
 *  files a defender as a midfielder, and "Defensive Midfielder" files him as a
 *  defender. Nothing errors; the badge is just wrong from then on.
 *
 *  Both return null for an unknown string rather than guessing, and callers
 *  keep the original when they get null - a coach's free text ("Libero") is
 *  not ours to overwrite. */
export function positionShort(label: string | null | undefined): string | null {
  const k = (label ?? "").trim().toLowerCase();
  return BY_LABEL.get(k)?.short ?? (BY_SHORT.has(k) ? BY_SHORT.get(k)!.short : null);
}

export function positionLabel(short: string | null | undefined): string | null {
  const k = (short ?? "").trim().toLowerCase();
  return BY_SHORT.get(k)?.label ?? (BY_LABEL.has(k) ? BY_LABEL.get(k)!.label : null);
}

/** Feet, exactly as the feed spells them - the panel prints this string. */
export const FEET = ["Right", "Left", "Both"] as const;
