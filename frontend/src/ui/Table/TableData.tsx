import type { FC, ReactNode, HTMLAttributes } from 'react';

interface Props extends HTMLAttributes<HTMLTableCellElement> {
  children?: ReactNode;
}

export const TableData: FC<Props> = ({ children, ...rest }) => {
  return (
    <td {...rest}>{children}</td>
  );
};