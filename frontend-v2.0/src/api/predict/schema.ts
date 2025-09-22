import { z } from 'zod';
import { BasePredictSchema } from '../baseSchema';

// PredictSchema
const PredictSchema = BasePredictSchema.extend({});
const PredictBatchSchema = z.object({
  results: z.array(PredictSchema),
  excel_file_path: z.string()
});

// Types
type PredictResponse = z.infer<typeof PredictSchema>;
type PredictBatchResponse = z.infer<typeof PredictBatchSchema>;

export {
  PredictSchema,
  PredictBatchSchema,
  type PredictResponse,
  type PredictBatchResponse
}
