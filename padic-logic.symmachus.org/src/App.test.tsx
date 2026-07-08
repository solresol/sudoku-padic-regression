import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("p-adic logic app", () => {
  it("compiles the sample CSP and exposes the search plan", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /compile evaluator/i }));

    expect(screen.getByText(/Ternary CNF/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Optimised p-adic linear regression solver/i)
    ).toBeInTheDocument();
    expect(screen.queryByText(/function evaluateMask/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Brute force/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Start exhaustive search/i })).toBeEnabled();
  });
});
