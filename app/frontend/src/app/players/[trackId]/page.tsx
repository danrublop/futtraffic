import { PlayerView } from "./player-view";

/** `params` is a Promise in this Next version — awaited here rather than read
 *  with useParams so the segment stays a server component and the client
 *  bundle only carries the view. */
export default async function PlayerPage(props: PageProps<"/players/[trackId]">) {
  const { trackId } = await props.params;
  return <PlayerView trackId={trackId} />;
}

/* The desktop build is a static export, which forces dynamicParams off:
 *  only the placeholder below exists there and desktop track links go to
 *  /track?t= instead (lib/nav.ts), so the placeholder is never served. Web
 *  builds render real ids on demand as before. ("_" is not a valid track id —
 *  track ids are integers — so it can never shadow a real one.) */
export async function generateStaticParams() {
  return [{ trackId: "_" }];
}
