import type { FC, ReactNode, HTMLAttributes } from 'react';

interface Props extends HTMLAttributes<HTMLTableCellElement> {
  children?: ReactNode;
}

export const TableHeader: FC<Props> = ({ children, ...rest }) => {
  return (
    <th {...rest}>{children}</th>
  );
};
