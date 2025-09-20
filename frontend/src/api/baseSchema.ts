import { z } from 'zod';

const BaseReportSchema = z.object({
  id: z.coerce.number(),
  title: z.string()
});

export {
  BaseReportSchema
};
