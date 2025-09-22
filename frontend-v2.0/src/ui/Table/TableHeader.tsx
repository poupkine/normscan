import type { FC, HTMLAttributes } from 'react';

interface Props extends HTMLAttributes<HTMLTableCellElement> { }

export const TableHeader: FC<Props> = ({ children, ...rest }) => {
  return (
    <th {...rest}>{children}</th>
  );
};
