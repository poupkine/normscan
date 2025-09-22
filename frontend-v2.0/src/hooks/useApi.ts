import { useState } from 'react';
import { CustomError } from '@services/error';

type Status = 'idle' | 'success' | 'error';

export function useApi<T extends (...args: any[]) => any>(fetchFn: T) {
  const [isFetching, setIsPending] = useState(false);
  const [status, setStatus] = useState<Status>('idle');
  const [data, setData] = useState<Awaited<ReturnType<typeof fetchFn>>>();
  const [error, setError] = useState<CustomError>();
  const controller = new AbortController();

  const sendRequest = async (
    ...args: Parameters<typeof fetchFn> extends [infer _, ...infer Rest] ? Rest : never
  ) => {
    if (status !== 'idle') {
      setStatus('idle');
    }
    if (error) {
      setError(undefined);
    }
    try {
      setIsPending(true);
      const responseData = await fetchFn(controller.signal, ...args);
      setIsPending(false);
      setStatus('success');
      setData(responseData);
    } catch (error) {
      setIsPending(false);
      if (!controller.signal.aborted) {
        setStatus('error');
        if (error instanceof CustomError) {
          setError(error);
        } else if (error instanceof Error) {
          setError(new CustomError(error));
        }
      } else {
        console.log(error)
      }
    }
  };

  const resetStatus = () => setStatus('idle');
  const resetError = () => {
    if (error) {
      setError(undefined);
    }
  };

  return {
    status,
    data,
    error,
    isFetching,
    sendRequest,
    resetStatus,
    resetError
  };
}