import { useState } from 'react';
import { type Alert, CustomError } from '@services/error';

export function useAlerts() {
  const [alerts, setAlerts] = useState<Alert[]>();

  const setAlert = (alert: Alert) => {
    setAlerts([alert]);
  };

  const clearAlerts = () => {
    if (alerts) {
      setAlerts(undefined);
    }
  };

  const parseAlerts = (error: CustomError) => {
    setAlerts(error.getAlerts());
  };

  return { alerts, setAlert, clearAlerts, parseAlerts };
}