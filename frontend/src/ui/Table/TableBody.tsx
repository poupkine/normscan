import type { FC, HTMLAttributes } from 'react';

interface Props extends HTMLAttributes<HTMLTableSectionElement> { }

export const TableBody: FC<Props> = ({ children, ...rest }) => {
  return (
    <tbody {...rest}>{children}</tbody>
  );
};
