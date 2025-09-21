import type { FC, ReactNode, HTMLAttributes } from 'react';

interface Props extends HTMLAttributes<HTMLTableSectionElement> {
  children?: ReactNode;
}

export const TableBody: FC<Props> = ({ children, ...rest }) => {
  return (
    <tbody {...rest}>{children}</tbody>
  );
};
