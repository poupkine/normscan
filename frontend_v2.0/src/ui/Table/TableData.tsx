import type { FC, HTMLAttributes } from 'react';

interface Props extends HTMLAttributes<HTMLTableCellElement> { }

export const TableData: FC<Props> = ({ children, ...rest }) => {
  return (
    <td {...rest}>{children}</td>
  );
};