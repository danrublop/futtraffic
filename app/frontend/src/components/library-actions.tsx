"use client";

import { useState } from "react";
import { Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { UploadCard } from "@/components/upload-card";
import { useMatches } from "@/hooks/use-matches";

/** GETTING A GAME IN — the one action the match library owned that could not
 *  go down with it.
 *
 *  Upload is step one of the whole product: a game goes in, the pipeline turns
 *  it into rows, and every surface reads those rows. It cannot live only on a
 *  page that no longer exists, and it cannot be buried inside a match picker
 *  that only appears once you already have a match.
 *
 *  IT LIVES IN THE SIDEBAR, reachable from every screen, as a grey filled
 *  button - at the foot, over the account row. It is the only control in
 *  the sidebar that puts something IN rather than taking you somewhere, so it
 *  does not wear the nav rows' look.
 *
 *  A search dialog used to sit beside it. It was a second way to find a match
 *  on a surface whose whole job is finding matches — the table below sorts by
 *  either club and filters by squad, season and opponent, which is strictly
 *  more than a name search could do.
 */
export function LibraryActions({ className }: { className?: string }) {
  const { refresh } = useMatches();
  const [adding, setAdding] = useState(false);

  return (
    <Dialog open={adding} onOpenChange={setAdding}>
      <DialogTrigger asChild>
        <Button variant="secondary" className={className}>
          <Upload className="size-4" /> Upload match
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Add a match</DialogTitle>
        </DialogHeader>
        <UploadCard
          onDone={() => {
            refresh();
            setAdding(false);
          }}
        />
      </DialogContent>
    </Dialog>
  );
}
