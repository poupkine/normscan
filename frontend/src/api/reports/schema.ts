import { z } from 'zod';
import { BaseReportSchema } from '../baseSchema';

// ResultSchema
const ReportSchema = BaseReportSchema.extend({});
const ReportListSchema = z.array(ReportSchema);

// Types
type ReportResponse = z.infer<typeof ReportSchema>;
type ReportListResponse = z.infer<typeof ReportListSchema>;

export {
  ReportSchema,
  ReportListSchema,
  type ReportResponse,
  type ReportListResponse
}
