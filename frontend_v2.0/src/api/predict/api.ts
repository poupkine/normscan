import {
  baseUrl,
  handleResponse,
  createRequestConfig,
} from '../utils';
import {
  PredictSchema,
  PredictBatchSchema,
  type PredictResponse,
  type PredictBatchResponse
} from './schema';

/**
 * API function `POST /predict`
 */
async function uploadFile(
  signal: AbortSignal,
  body: FormData
): Promise<PredictResponse> {
  return fetch(`${baseUrl}/predict`, {
    ...createRequestConfig('POST'),
    signal,
    body
  })
    .then(handleResponse)
    .then(response => response.json())
    .then(data => PredictSchema.parse(data));
}

/**
 * API function `POST /batch_predict`
 */
async function uploadFileList(
  signal: AbortSignal,
  body: FormData
): Promise<PredictBatchResponse> {
  return fetch(`${baseUrl}/batch_predict`, {
    ...createRequestConfig('POST'),
    signal,
    body
  })
    .then(handleResponse)
    .then(response => response.json())
    .then(data => PredictBatchSchema.parse(data));
}

export { uploadFile, uploadFileList };
