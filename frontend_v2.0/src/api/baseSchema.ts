import { z } from 'zod';

const BasePredictSchema = z.object({
  filename: z.string(),
  processing_status: z.string(),
  error_detail: z.string().nullable(),
  probability_of_pathology: z.number(),
  study_uid: z.string(),
  series_uid: z.string(),
  processing_time_sec: z.number(),
  path_to_study: z.string(),
  time_of_processing: z.number(),
  pathology: z.number()
});

export {
  BasePredictSchema
};
