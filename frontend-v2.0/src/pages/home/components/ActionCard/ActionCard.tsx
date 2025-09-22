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


const FormSchema = z.object({ file: ZipFileFormSchema });
type FormSchema = z.infer<typeof FormSchema>;

export const ActionCard: FC = () => {
  const dispatch = useAppDispatch();
  const [isLoadFormOpen, setIsLoadFormOpen] = useState(false);
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
    formState: { errors }
  } = useForm<FormSchema>({
    resolver: zodResolver(FormSchema)
  });

  useEffect(() => {
    if (fileStatus === 'success' && fileData) {
      console.log(fileData);
      dispatch(setResultList([fileData]));
    } else if (fileStatus === 'error' && fileError) {
      dispatch(setErrorMessage(fileError.getErrorMessage()))
    }

    if (fileListStatus === 'success' && fileListData) {
      console.log(fileListData);
      dispatch(setResultList(fileListData.results));
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
          Поддерживаются форматы DICOM, ZIP
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
          <input type="file" multiple accept=".zip" {...register('file')} />
          {errors.file && <p className="error">{errors.file.message}</p>}
          <DialogButton type="submit">Загрузить</DialogButton>
        </DialogForm>
      </Dialog>
    </Card>
  );
};
