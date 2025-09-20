import { createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { RootState } from '../store';
import type { ErrorMessage } from '@services/error';

export interface ErrorState {
  message: ErrorMessage | undefined;
}

const initialState: ErrorState = {
  message: undefined
};

export const errorSlice = createSlice({
  name: 'error',
  initialState,
  reducers: {
    setErrorMessage: (
      state,
      action: PayloadAction<ErrorState['message']>
    ) => {
      state.message = action.payload;
    },
    resetError: () => initialState
  }
});

export const { setErrorMessage, resetError } = errorSlice.actions;
export const selectError = (state: RootState) => state.error;

export default errorSlice.reducer;