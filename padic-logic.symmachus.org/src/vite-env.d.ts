/// <reference types="vite/client" />

declare global {
  interface LanguageModelPrompt {
    role: "system" | "user" | "assistant";
    content: string;
    prefix?: boolean;
  }

  interface LanguageModelSession {
    prompt(
      input: string | LanguageModelPrompt[],
      options?: LanguageModelPromptOptions
    ): Promise<string>;
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
    signal?: AbortSignal;
  }

  interface LanguageModelPromptOptions {
    signal?: AbortSignal;
  }

  var LanguageModel: LanguageModelFactory | undefined;
  var languageModel: LanguageModelFactory | undefined;
}

export {};
