// Typed errors for the Laya JS runtime.
// All carry `cause` so CLI/worker can print the root failure.
export class LayaConfigError extends Error {
  constructor(message: string, opts?: ErrorOptions) {
    super(message, opts);
    this.name = 'LayaConfigError';
  }
}

export class LayaEncodeError extends Error {
  constructor(message: string, opts?: ErrorOptions) {
    super(message, opts);
    this.name = 'LayaEncodeError';
  }
}

export class LayaInferenceError extends Error {
  constructor(message: string, opts?: ErrorOptions) {
    super(message, opts);
    this.name = 'LayaInferenceError';
  }
}
