export interface Alert {
  status: 'success' | 'error';
  message: string;
}

export interface ErrorMessage {
  title: string;
  text: string;
}

interface NonFieldError {
  non_fieled_errors?: string[];
}

interface DetailError {
  detail?: string;
}

interface FieldError {
  [key: string]: string[];
}

type ResponseError = (NonFieldError & DetailError & FieldError);

export class CustomError {
  response: Response | undefined;
  error: Error | undefined;
  data: undefined | ResponseError;
  defaultMessage: string = 'Неизвестная ошибка';

  constructor(obj: Response | Error, data?: ResponseError) {
    this.data = data;
    if (obj instanceof Response) {
      this.response = obj;
    } else if (obj instanceof Error) {
      this.error = obj;
    }
  }

  getAlerts(): Alert[] {
    if (this.response && this.data) {
      return this.parseAlerts();
    } else if (this.error) {
      return [{
        status: 'error',
        message: `${this.error.name} ${this.error.message}`
      }];
    }
    return [{
      status: 'error',
      message: 'Детали ошибки отсутствуют в CustomError.getAlerts()'
    }];
  }

  parseAlerts(): Alert[] {
    switch (this.response && this.response.status) {
      case 401:
      case 403:
        if (this.data?.detail) {
          return [{
            status: 'error',
            message:
              detailMap[this.data.detail] ||
              this.data.detail ||
              this.defaultMessage,
          }];
        } else if (Array.isArray(this.data?.detail)) {
          return [{
            status: 'error',
            message:
              this.data.detail.toString()
          }];
        } else {
          return [{
            status: 'error',
            message:
              this.defaultMessage
          }];
        }
      case 404:
        return [{
          status: 'error',
          message: responseStatusCodeMap[404]
        }];
      case 400:
        if (this.data?.non_field_errors) {
          return this.data.non_field_errors?.reduce(
            (acc: Alert[], value: string) => {
              acc.push({
                status: 'error',
                message: nonFieldErrorMap[value] || value || this.defaultMessage
              });
              return acc;
            }, []);
        } else if (this.data?.detail) {
          return [{
            status: 'error',
            message:
              detailMap[this.data.detail] ||
              this.data.detail ||
              this.defaultMessage,
          }];
        } else if (this.data) {
          return Object.entries(this.data)
            .reduce<Alert[]>((acc, [key, value]) => {
              if (Array.isArray(value) && key !== 'protected_objects') {
                acc.push(...value.map((v): Alert => {
                  const k = fieldMap[key] || key; // k: fieldName
                  let message = detailMap[v] || v || this.defaultMessage;
                  if (k) {
                    message = `${k}: ${message}`;
                  }
                  return { status: 'error', message: message };
                }));
              }
              return acc;
            }, []);
        } else {
          return [{
            status: 'error',
            message: 'Не удалось получить данные ошибки в parseAlerts()'
          }];
        }
      case 500:
        return [{
          status: 'error',
          message: responseStatusCodeMap[500]
        }];
      default:
        return [{ status: 'error', message: this.defaultMessage }];
    }
  }

  getErrorMessage(): ErrorMessage {
    if (this.response) {
      return {
        title: `Ошибка ${this.response.status}`,
        text: responseStatusCodeMap[this.response.status] || this.defaultMessage
      };
    } else if (this.error) {
      return {
        title: `Программная ошибка ${this.error.name}`,
        text: ErrorMessage[this.error.message] || this.error.message
      };
    } else {
      return {
        title: 'Программная ошибка',
        text: this.defaultMessage
      };
    }
  }
}

const ErrorMessage: { [key: string]: string } = {};

const nonFieldErrorMap: { [key: string]: string } = {};

const detailMap: { [key: string]: string } = {};

const fieldMap: { [key: string]: string } = {};

const responseStatusCodeMap: { [key: number]: string } = {
  400: 'Некорректный запрос',
  401: 'Требуется аутентификация',
  403: 'Запрещено',
  404: 'Ресурс не найден',
  408: 'Тайм-аут запроса',
  429: 'Слишком много запросов',
  500: 'Внутренняя ошибка сервера',
  501: 'Не реализовано',
  502: 'Плохой шлюз',
  503: 'Служба недоступна',
  504: 'Тайм-аут шлюза',
};