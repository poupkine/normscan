import { createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { RootState } from '@store/store';
import type { PredictResponse } from '@api/predict';

export interface HomePageState {
  resultList: PredictResponse[];
}

const initialState: HomePageState = {
  resultList: []
};

export const homePageSlice = createSlice({
  name: 'pages/homePage',
  initialState,
  reducers: {
    setResultList: (state, action: PayloadAction<PredictResponse[]>) => {
      state.resultList = action.payload;
    },
    reset: () => initialState
  }
});

export const { setResultList, reset } = homePageSlice.actions;
export const selectResultList = (state: RootState) => state.pages.homePage.resultList;

export default homePageSlice.reducer;