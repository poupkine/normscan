import type { FC, HTMLAttributes } from 'react';

interface Props extends HTMLAttributes<HTMLTableSectionElement> { }

export const TableHead: FC<Props> = ({ children, ...rest }) => {
  return (
    <thead {...rest}>{children}</thead>
  );
};
