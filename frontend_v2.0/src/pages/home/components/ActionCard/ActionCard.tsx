import { type FC, useState, useEffect } from 'react';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import { useAppDispatch } from '@store/hooks';
import { setErrorMessage } from '@store/slices/errorSlice';
import { setResultList } from '@pages/home/slice';
import { useApi } from '@hooks/useApi';
import { uploadFile, uploadFileList } from '@api/predict';
import { ZipFileFormSchema } from '@api/baseFormSchema';
import { Card } from '@ui/Card';
import { Dialog, DialogForm, DialogButton } from '@ui/Dialog';
import spriteUrl from '@assets/sprite.svg';
import styles from './ActionCard.module.css';

const MESSAGES = {
  FILE_SUCCESS: 'Файл обработан, можете посмотреть результат в таблице.',
  FILE_LIST_SUCCESS: 'Файлы обработаны, можете посмотреть результаты в таблице и скачать отчет по ссылке.'
};


const FormSchema = z.object({ file: ZipFileFormSchema });
type FormSchema = z.infer<typeof FormSchema>;

export const ActionCard: FC = () => {
  const dispatch = useAppDispatch();
  const [isLoadFormOpen, setIsLoadFormOpen] = useState(false);
  const [successMessage, setSuccessMessage] = useState<string>();
  const {
    status: fileStatus,
    data: fileData,
    error: fileError,
    isFetching: isFileFetching,
    sendRequest: sendFile
  } = useApi(uploadFile);
  const {
    status: fileListStatus,
    data: fileListData,
    error: fileListError,
    isFetching: isFileListFetching,
    sendRequest: sendFileList
  } = useApi(uploadFileList);
  const { } = useApi(uploadFileList);
  const {
    register,
    reset,
    handleSubmit,
    watch,
    formState: { errors }
  } = useForm<FormSchema>({
    resolver: zodResolver(FormSchema)
  });
  const file = watch('file');

  useEffect(() => {
    if (fileStatus === 'success' && fileData) {
      dispatch(setResultList([fileData]));
      reset();
      setSuccessMessage(MESSAGES.FILE_SUCCESS);
    } else if (fileStatus === 'error' && fileError) {
      dispatch(setErrorMessage(fileError.getErrorMessage()))
    }

    if (fileListStatus === 'success' && fileListData) {
      dispatch(setResultList(fileListData.results));
      reset();
      setSuccessMessage(MESSAGES.FILE_LIST_SUCCESS);
    } else if (fileListStatus === 'error' && fileListError) {
      dispatch(setErrorMessage(fileListError.getErrorMessage()))
    }

  },
    [
      fileStatus,
      fileListStatus,
      fileData,
      fileListData,
      fileError,
      fileListError
    ]);

  const onSubmit = handleSubmit((formData: FormSchema) => {
    const newFormData = new FormData();
    if (formData.file.length > 1) {
      [...formData.file].forEach(file => {
        newFormData.append('files', file);
      });
      sendFileList(newFormData);
    } else {
      newFormData.append('file', formData.file[0]);
      sendFile(newFormData);
    }
  });

  const resetAndClose = () => {
    reset();
    setIsLoadFormOpen(false);
    setSuccessMessage(undefined);
  };

  return (
    <Card className={styles['action-card']}>
      <h2 className='visually-hidden'>Загрузить КТ-исследование</h2>
      <svg className={styles['action-card__icon']} width="300" height="224" aria-hidden="true">
        <use xlinkHref={`${spriteUrl}#icon-cloud-upload-big`}></use>
      </svg>
      <button
        className={`btn ${styles['action-card__button']}`}
        onClick={() => setIsLoadFormOpen(true)}
      >
        Загрузить КТ-исследование
      </button>
      <div className={styles['action-card__info']}>
        <span className={styles['action-card__info-text']}>
          Поддерживается формат DICOM в ZIP архиве
        </span>
        <span className={styles['action-card__info-text']}>
          Максимальный размер файла 1ГБ
        </span>
        <span className={styles['action-card__info-text']}>
          Исследование будет автоматизировано
        </span>
      </div>
      <Dialog
        title="Загрузить КТ"
        isOpen={isLoadFormOpen}
        handleClose={resetAndClose}
        isFetching={(isFileFetching || isFileListFetching)}
      >
        <DialogForm
          onSubmit={onSubmit}
        >
          <label
            className={styles['action-card__input-label']}
          >
            <input
              className={styles['action-card__input-field']}
              type="file"
              multiple accept=".zip"
              title="Перенесите или загрузите нужный файл"
              {...register('file')}
              disabled={file?.length > 0}
            />
            <svg
              className={file?.length > 0
                ? `${styles['action-card__input-icon']} ${styles['action-card__input-icon--invisible']}`
                : styles['action-card__input-icon']
              }
              width="300"
              height="224"
              aria-hidden="true"
            >
              <use xlinkHref={`${spriteUrl}#icon-cloud-upload-big`}></use>
            </svg>
            <span className={file?.length > 0
              ? `${styles['action-card__input-text']} ${styles['action-card__input-text--invisible']}`
              : styles['action-card__input-text']
            }>
              Перенесите или загрузите нужный файл
            </span>
            {file?.length > 0 &&
              <ul className={styles['action-card__input-file-list']}>
                {[...file].map(file => (
                  <li
                    key={file.name}
                    className={styles['action-card__input-file-list-item']}
                  >
                    {file.name}
                  </li>
                ))}
              </ul>
            }
          </label>
          {errors.file && <p className={styles['action-card__error']}>{errors.file.message}</p>}
          {successMessage && <p className={styles['action-card__success']}>{successMessage}</p>}
          <DialogButton
            type="submit"
            disabled={isFileFetching || isFileListFetching || Boolean(successMessage)}
          >
            Загрузить
          </DialogButton>
        </DialogForm>
      </Dialog>
    </Card>
  );
};
