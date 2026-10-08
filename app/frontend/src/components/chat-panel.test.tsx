import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { AnswerText } from "./chat-panel";

/** Agent answers are markdown from the model. Two things must hold: what the
 *  model formats reads as formatting (a comparison table was a wall of pipes),
 *  and what it writes as HTML never becomes markup. */
describe("AnswerText", () => {
  it("renders a comparison table, a list and bold", () => {
    const { container } = render(
      <AnswerText
        text={"**Provisional** figures:\n\n- ranks survive\n- absolutes do not\n\n| # | km |\n|---|---|\n| 7 | 10.2 |\n| 11 | 9.8 |"}
      />,
    );
    expect(container.querySelector("strong")?.textContent).toBe("Provisional");
    expect(container.querySelectorAll("li")).toHaveLength(2);
    expect(container.querySelectorAll("tbody tr")).toHaveLength(2);
    expect(container.textContent).not.toContain("|");
    // react-markdown passes each component its syntax-tree node; spread onto
    // the element it became node="[object Object]" on every tag.
    expect(container.innerHTML).not.toContain("node=");
  });

  it("shows raw HTML in a reply as text, never as elements", () => {
    const { container } = render(
      <AnswerText text={'Look: <img src=x onerror="alert(1)"> and <script>alert(2)</script>'} />,
    );
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("script")).toBeNull();
  });

  it("does not throw on half-streamed markdown", () => {
    const { container } = render(<AnswerText text={"| # | km |\n|---|--"} />);
    expect(container.textContent).toContain("km");
  });
});
