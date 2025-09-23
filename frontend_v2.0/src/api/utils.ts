import { CustomError } from '@services/error';

export const baseUrl = import.meta.env.VITE_API_URL;

export type Id = string | number;
export type Method = 'GET' | 'POST' | 'PATCH' | 'DELETE';
export type ContentType = 'application/json' | '';

/**
 * Helper function to create fetch() request config.
 */
export function createRequestConfig(
  method: Method,
  contentType?: ContentType
) {
  const headers: HeadersInit = {};

  if (contentType) {
    headers['Content-Type'] = contentType;
  }

  if (method !== 'GET') {
    const cookies = Object.fromEntries(
      document.cookie.split('; ').map(x => x.split('='))
    );
    headers['X-CSRFToken'] = cookies?.csrftoken;
  }

  return {
    method,
    headers,
    // credentials: 'include'
  } as RequestInit;
}

/**
 * Helper function to create url with search params.
 */
export function createURL(url: string, params?: URLSearchParams) {
  if (params) {
    return `${url}?${params}`;
  }
  return url;
}

/**
 * Response handler for `fetch()` response.
 */
export async function handleResponse(response: Response): Promise<Response> {
  if (!response.ok) {
    let data = undefined;
    try {
      data = await response.json();
    } catch (error) {
      console.log(error);
    }
    console.log(response, data);
    throw new CustomError(response, data);
  }
  return response;
}
