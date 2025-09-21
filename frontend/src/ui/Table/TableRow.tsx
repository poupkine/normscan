import type { FC, ReactNode, HTMLAttributes } from 'react';

interface Props extends HTMLAttributes<HTMLTableRowElement> {
  children?: ReactNode;
}

export const TableRow: FC<Props> = ({ children, ...rest }) => {
  return (
    <tr {...rest}>{children}</tr>
  )
}
