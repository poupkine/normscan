import { z } from 'zod';

const BasePredictSchema = z.object({
  path_to_study: z.string(),
  study_uid: z.string(),
  series_uid: z.string(),
  probability_of_pathology: z.number(),
  pathology: z.number(),
  processing_status: z.string(),
  time_of_processing: z.number(),
  most_dangerous_pathology_type: z.string().nullable(),
  pathology_localization: z.string().nullable()
});

export {
  BasePredictSchema
};
