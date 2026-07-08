/// <reference types="vite/client" />

declare global {
  interface LanguageModelPrompt {
    role: "system" | "user" | "assistant";
    content: string;
    prefix?: boolean;
  }

  interface LanguageModelSession {
    prompt(input: string | LanguageModelPrompt[]): Promise<string>;
    destroy?: () => void;
  }

  interface LanguageModelFactory {
    availability(options?: LanguageModelCreateOptions): Promise<
      "unavailable" | "downloadable" | "downloading" | "available"
    >;
    create(options?: LanguageModelCreateOptions): Promise<LanguageModelSession>;
  }

  interface LanguageModelCreateOptions {
    expectedInputs?: Array<{ type: "text"; languages: string[] }>;
    expectedOutputs?: Array<{ type: "text"; languages: string[] }>;
    initialPrompts?: LanguageModelPrompt[];
    monitor?: (monitor: EventTarget) => void;
  }

  var LanguageModel: LanguageModelFactory | undefined;
}

export {};
