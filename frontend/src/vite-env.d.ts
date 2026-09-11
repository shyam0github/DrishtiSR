/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "false" switches src/api/client.ts to the live backend. Anything else, or unset, is MOCK_MODE. */
  readonly VITE_MOCK_MODE?: string;
  /** Live backend base URL. Defaults to "/api". */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
