interface Email {
  value: string; // Enforce email format validation
}

interface SecureString {
  value: string; // Enforce password length, character requirements, etc.
}

interface HashedPassword {
  algorithm: string;
  value: string;
}

function login(credentials: { username: Email; password: SecureString }): boolean {
  const hashedPassword: HashedPassword = secureHashPassword(credentials.password);
  // ... (code with explicit type checks for security-sensitive operations)
}

// Improved Code

interface Email {
  value: string; // Enforce email format validation
}

interface SecureString {
  value: string; // Enforce password length, character requirements, etc.
  toString(): string; // Convert SecureString to normal string when necessary
  clear(): void; // Clear the SecureString after use
}

interface HashedPassword {
  algorithm: string;
  value: string;
}

function login(credentials: { username: Email; password: SecureString }): boolean {
  const hashedPassword: HashedPassword = secureHashPassword(credentials.password);
  // ... (code with explicit type checks for security-sensitive operations)
  credentials.password.clear(); // Clearing the SecureString after use
}
