// The renderer is a plain web app served over futtraffic:// - it needs almost
// nothing from the main process. This bridge exists so the UI can tell it is
// running inside the desktop shell (titlebar padding for the traffic lights,
// download affordances) without sniffing the user agent.
import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("futtrafficDesktop", {
  isElectron: true,
  platform: process.platform,
  /** A finished document in, PDF bytes out - see "print-pdf" in main.ts. */
  printPdf: (html: string): Promise<Uint8Array> => ipcRenderer.invoke("print-pdf", html),
});
