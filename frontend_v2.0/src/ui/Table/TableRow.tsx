import type { FC, HTMLAttributes } from 'react';

interface Props extends HTMLAttributes<HTMLTableRowElement> { }

export const TableRow: FC<Props> = ({ children, ...rest }) => {
  return (
    <tr {...rest}>{children}</tr>
  )
}
