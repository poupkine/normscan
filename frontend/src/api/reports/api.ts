import {
  baseUrl,
  handleResponse,
  createRequestConfig,
} from '../utils';
import {
  ReportSchema,
  ReportListSchema,
  type ReportResponse,
  type ReportListResponse
} from './schema';

/**
 * API function `POST /reports`
 */
async function loadFile(
  signal: AbortSignal,
  body: FormData
): Promise<ReportResponse> {
  return fetch(`${baseUrl}/reports`, {
    ...createRequestConfig('POST'),
    signal,
    body
  })
    .then(handleResponse)
    .then(response => response.json())
    .then(data => ReportSchema.parse(data));
}

/**
 * API function `GET /reports`
 */
async function getReportList(
  signal: AbortSignal,
): Promise<ReportListResponse> {
  return fetch(`${baseUrl}/reports`, {
    ...createRequestConfig('GET'),
    signal
  })
    .then(handleResponse)
    .then(response => response.json())
    .then(data => ReportListSchema.parse(data));
}

export { loadFile, getReportList };
