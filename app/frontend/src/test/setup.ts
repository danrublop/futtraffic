import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

/** What every jsdom test needs and none of them should have to say.
 *
 *  Testing Library renders into a container it appends to document.body and
 *  does not remove it. Without this, the second test in a file queries a
 *  document holding the first test's markup too — which does not fail, it
 *  returns the WRONG element, and the suite passes while asserting nothing.
 */
afterEach(cleanup);
