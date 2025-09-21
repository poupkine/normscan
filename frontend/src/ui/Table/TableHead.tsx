import type { FC, ReactNode, HTMLAttributes } from 'react';

interface Props extends HTMLAttributes<HTMLTableSectionElement> {
  children?: ReactNode;
}

export const TableHead: FC<Props> = ({ children, ...rest }) => {
  return (
    <thead {...rest}>{children}</thead>
  );
};
