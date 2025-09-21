class StorageService {
  private prefix: string;

  constructor(prefix: string = 'app_') {
    this.prefix = prefix;
  }

  public get<T>(key: string): T | null {
    const data = localStorage.getItem(this.prefix + key);
    return data ? JSON.parse(data) as T : null;
  }

  public set(key: string, value: unknown): void {
    localStorage.setItem(this.prefix + key, JSON.stringify(value));
  }

  public remove(key: string): void {
    localStorage.removeItem(this.prefix + key);
  }

  public clear(): void {
    Object.keys(localStorage)
      .filter((key) => key.startsWith(this.prefix))
      .forEach((key) => localStorage.removeItem(key));
  }
}

export const appLocalStorage = new StorageService();