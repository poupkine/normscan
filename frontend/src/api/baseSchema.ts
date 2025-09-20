import { z } from 'zod';

const BaseReportSchema = z.object({
  id: z.number(),
});

export {
  BaseReportSchema
};
