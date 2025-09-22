import { z } from 'zod';

// Zip file form schema
const fileSizeLimit = 5 * 1024 * 1024; // 5MB
const maxFileSizeMessage = 'Максимальный размер файла 5MB';
const documentMimeTypes = [
  'application/zip',
  'application/x-zip-compressed',
  'multipart/x-zip'
];

const ZipFileFormSchema = z
  .instanceof(FileList, { message: 'Выберите файл' })
  .refine((list) => list.length > 0, { message: 'Выберите файл' })
  .refine((fileList) => {
    // Check all files (MIME type)
    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      if (!documentMimeTypes.includes(file.type)) {
        return false;
      }
    }
    return true;
  }, { message: 'Разрешенные форматы .zip' })
  .refine((fileList) => {
    // Check all files (size)
    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      if (file.size > fileSizeLimit) {
        return false;
      }
    }
    return true;
  }, { message: maxFileSizeMessage });

export {
  ZipFileFormSchema
};