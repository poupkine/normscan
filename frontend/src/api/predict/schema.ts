import { z } from 'zod';
import { BasePredictSchema } from '../baseSchema';

// PredictSchema
const PredictSchema = BasePredictSchema.extend({});
const PredictBatchSchema = z.object({
  results: z.array(PredictSchema),
  report_available: z.boolean()
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
