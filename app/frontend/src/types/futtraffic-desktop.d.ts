export {};

/* The desktop shell (app/desktop/electron/preload.ts) exposes this on the window
 * when the app runs inside Electron. Guarded everywhere it's read so the web
 * build and SSR never touch it. */
declare global {
  interface Window {
    futtrafficDesktop?: {
      isElectron: boolean;
      platform: string;
    };
  }
}
