import pandas as pd
import shap
from catboost import CatBoostRegressor

from interpretability.data import df


class Explainer:
    def __init__(self, model, df):
        self.model = model
        self.df = df

    def density_plot(self, column_name: str) -> None:
        """Выводит график распределения значений в колонке.

        Keyword arguments:
        column_name -- имя колонки
        """

        sns.histplot(self.df[column_name], kde=True, stat="density",
                     bins=36, color='darkblue')
        return None

    def joint_destribution(self, column_name_1: str, column_name_2: str) -> None:
        """Выводит график совместного распределения двух колонок.

        Keyword arguments:
        column_name_1 -- имя колонки 1 (ось x)
        column_name_2 -- имя колонки 2 (ось y)
        """
        sns.jointplot(data=self.df, x=column_name_1, y=column_name_2, kind='kde')
        return None

    def variable_effect(self, column_name: str, ts: float, move_dir: int) -> str:
        """Функция для исследования эффекта фичи на таргет.

        Keyword arguments:
        column_name -- целевая колонка, по которой мы исследуем эффект
        ts -- доля изменения значений колонки
        move_dir -- определяет, уменьшаем или увеличиваем значение колонки (значение 1 или -1)

        Output:
        Изменение таргета в процентах
        """

        if move_dir not in [1, -1]:
            raise ValueError("flag может быть только 1 или -1")

        data = self.df.copy()
        preds = self.model.predict(data)
        data[column_name] *= 1 + move_dir * ts
        new_preds = self.model.predict(data)
        # return [(j - i) / i * 100 for i, j in zip(preds, new_preds)]
        return f'{preds.sum() / new_preds.sum() - 1:2%}'

    def compare_classes(self, column_name: str) -> None:
        """Разница распределений целевой колонки по защищенному классу.

        Keyword arguments:
        column_name -- целевая колонка
        """
        sns.displot(self.df, x=column_name, hue='Sex', label='Пол', kind='kde')
        return None

    def accept_percentage(self, column_name: str, plot: bool) -> str:
        """Процент положительного класса в задаче категоризации по защищенным классам

        Keyword arguments:
        column_name (str) -- колонка защищенных классов
        plot (bool) -- рисовка графика

        Output:
        строка с процентом
        """

        classes = self.df[column_name].unique()
        acceptance_list = list()
        ans = ""
        for cls in classes:
            acceptance = len(self.df[(self.df[column_name] == cls) & (self.df['descrete_score'] == 1)]) / len(
                self.df[self.df[column_name] == cls])
            acceptance_list.append(len(self.df[(self.df[column_name] == cls) & (self.df['descrete_score'] == 1)]) / len(
                self.df[self.df[column_name] == cls]))
            ans += f"Процент одобренных среди {cls} = {acceptance:2%}, отклоненных - {1 - acceptance:2%}\n"
        if plot:
            plt.bar(classes, acceptance_list)
        plt.legend()
        plt.show()
        return ans

    def tpr_fpr(self, column_name: str) -> None:
        """Анализ fpr, tpr по защищенному классу.
        Также вычисляет fpr и tpr в значении threshold.

        Keyword arguments:
        column_name (str) -- колонка защищенного класса
        """

        classes = self.df[column_name].unique()
        #   _, ax = plt.subplots(figsize=(6, 24), nrows=3, ncols=1)
        #   color = iter(plt.get_cmap('rainbow')(np.linspace(0,1,len(classes))))
        #   graphs_fpr = list()
        #   graphs_tpr = list()
        for cls in classes:
            #   fpr, tpr, _ = roc_curve(df[df['Sex'] == cls]['target'], df[df['Sex'] == cls]['score'])
            #   sns.regplot(x=fpr, y=tpr,label=cls)
            fpr, tpr, thresholds = roc_curve(self.df[self.df[column_name] == cls]['target'],
                                             self.df[self.df[column_name] == cls]['score'])
            i = np.arange(len(tpr))  # index for df
            roc = pd.DataFrame(
                {'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i), '1-fpr': pd.Series(1 - fpr, index=i),
                 'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(thresholds, index=i)})
            roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

            # Plot tpr vs 1-fpr
            fig, ax = pl.subplots()
            pl.plot(roc['tpr'])
            pl.plot(roc['1-fpr'], color='red')
            pl.xlabel('1-False Positive Rate')
            pl.ylabel('True Positive Rate')
            pl.title('Receiver operating characteristic')
            pl.text(0, 0, f'')
            ax.set_xticklabels([])
            ans = f"Metrics for {cls} on cutoff\n"
            for col in roc_t.columns:
                ans += f"{col}: {roc_t[col].iloc[0]}\n"
            print(ans)
            """
            line1, = ax[1].plot(fpr, np.linspace(0,len(fpr),len(fpr)), label=f'FPR для {column_name} : {cls}')
            graphs_fpr.append(line1)
            ax[1].legend(handles=graphs_fpr)
            line2, = ax[2].plot(tpr, np.linspace(0,len(fpr),len(fpr)), label=f'TPR для {column_name} : {cls}')
            graphs_tpr.append(line2)
            ax[2].legend(handles=graphs_tpr)

            RocCurveDisplay.from_predictions(
                df[df[column_name] == cls]['target'],
                df[df[column_name] == cls]['score'],
                color=next(color),
                ax=ax[0],
                name=f"ROC curve for {cls}"
            )

        _ = ax[0].set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="ROC curves:\n",
        ) 
        _ = ax[1].set(
            xlabel="Occasion",
            ylabel="False Positive Rate",
            title="FPR\n",
        )  
        _ = ax[2].set(
            xlabel="Occasions",
            ylabel="True Positive Rate",
            title="TPR\n"
        )      """
        return None

    def shapper(self, ind: int) -> None:
        """График вклада строчки в таргет

        Keyword_arguments:
        ind (int) -- индекс строчки
        """

        explainer = shap.Explainer(self.model)
        exp = explainer(self.df)
        s = exp.values[ind].sum()
        for i in range(exp.values[ind].size):
            exp.values[ind][i] /= s
        exp = shap.Explanation(exp, exp.base_values, self.df, feature_names=self.df.columns)
        shap.plots.waterfall(exp[ind], max_display=8)
        return exp[ind]

    def get_shap_values(self, ind):
        explainer = shap.Explainer(self.model)
        exp = explainer(self.df)
        s = exp.values[ind].sum()
        for i in range(exp.values[ind].size):
            exp.values[ind][i] /= s
        exp = shap.Explanation(exp, exp.base_values, self.df, feature_names=self.df.columns)
        return exp[ind]

    def difference_percentage(self, column_name: str, target_col: str) -> str:
        """Функция вычисления разницы процентов одобренных защищенных классов,
        разницы процентов отклоненных защищенных классов

        Keyword arguments:
        df -- pd.Dataframe
        column_name -- название колонки защищенного класса
        target_col -- название колонки такгета

        Output: str
        """

        classes = self.df[column_name].unique()
        accept_percentage = 0
        decline_percentage = 0
        if len(classes) > 2:
            return "Разницы не существует"
        for cls in classes:
            accept_percentage = len(self.df[(self.df[column_name] == cls) & (self.df[target_col] == 1)].index) / len(
                self.df[self.df[column_name] == cls].index) - accept_percentage
            decline_percentage = len(self.df[(self.df[column_name] == cls) & (self.df[target_col] == 0)].index) / len(
                self.df[self.df[column_name] == cls].index) - decline_percentage
        return (
            f'Разница в проценте принятния между {classes[1]} и {classes[0]} составляет {accept_percentage:.2%}\n'
            f'Разница в проценте отклонения между {classes[1]} и {classes[0]} составляет {decline_percentage:.2%}\n'
        )

    def search(self, target_col: str) -> pd.DataFrame:
        """Поиск двух наиболее близких строчек в df с разным результатом задачи категоризации

        Keyword arguments:
        target_col (str) -- колонка, где указан таргет

        Output:
        df из двуз строчек
        """

        num_columns = self.df._get_numeric_data().columns
        df_copy = self.df.copy()
        df_copy[num_columns] = self.df[num_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        accept_df = df_copy[df_copy[target_col] == 1]._get_numeric_data()
        decline_df = df_copy[df_copy[target_col] == 0]._get_numeric_data()
        dist = cdist(accept_df, decline_df, 'euclid')
        arg = np.argwhere(dist == np.min(dist))
        return self.df.loc[[accept_df.index[arg[0][0]], decline_df.index[arg[0][1]]]]

    def distribution(self, column_name: str, protected_class_column: str) -> None:
        """Функция рисующая графики распределения по защищенному классу

        Keyword arguments:
        column_name (str) -- колонка значений
        protected_class_column (str) -- колонка защищенного класса
        """

        if is_numeric_dtype(self.df[column_name]):
            axes = self.df.hist([column_name], by=protected_class_column, legend=True)
        else:
            groups = self.df[protected_class_column].unique()
            fig, axes = plt.subplots(nrows=1, ncols=len(groups))  # TODO: Добавить крассивую группировку
            for idx, group in enumerate(groups):
                self.df.groupby(protected_class_column)[column_name].value_counts()[group].plot(kind='bar',
                                                                                                ax=axes[idx])
        return None

    def result_of_centroid(self, target_col) -> int:
        """Вывод результата самого \"среднего\" результата в датасете

        Keyword arguments:
        target_col (str) -- колонка таргета
        """

        df_copy = self.df.copy()
        num_columns = self.df._get_numeric_data().columns
        df_copy = df_copy[num_columns]
        return self.df.iloc[[df_copy.sub(df_copy.mean()).pow(2).sum(1).idxmin()]][target_col].iloc[0]

    def mean_protected(self, column_name: str, protected_class_column: str) -> str:
        """Среднее значение колонки по защищенному классу

        Keyword arguments:
        column_name (str) -- целевая колонка, по которой надо расчитать среднее
        protected_class_column (str) -- колонка защищенного класса
        """

        data = self.df.groupby(protected_class_column)[column_name].mean()
        ans = ''
        for i, v in data.items():
            ans += f'Среденее значение колонки {column_name} для класса {i} = {v}\n'
        return ans

    def mean_shapper(self, protected_class_column: str) -> None:
        """Средний вклад каждой колонки по всему df

        Keyword arguments:
        protected_class_column (str) -- колонка защищенного класса
        """

        explainer = shap.Explainer(self.model)
        for cls in self.df[protected_class_column].unique():
            exp = explainer(self.df[self.df[protected_class_column] == cls])
            exp = shap.Explanation(exp, exp.base_values, self.df[self.df[protected_class_column] == cls],
                                   feature_names=self.df.columns)
            exp.values = exp.values.mean(axis=0)
            exp.base_values = exp.base_values[0]
            exp.data = exp.data[0]
            shap.plots.waterfall(exp, max_display=8)
        return None

    def mean_target(self, protected_class_column: str, target_col: str, true_target_col: str) -> str:
        """Среднее значение таргета по классу.

        Keyword arguments:
        protected_class_column (str) -- колонка защищенного класса
        target_col (str) -- колонка таргета
        true_target_col (str) -- колонка с истинными значениями таргета
        """

        ans = ""
        true_vals = list()
        model_vals = list()
        cats = self.df[protected_class_column].unique()
        for cls in self.df[protected_class_column].unique():
            model_val = self.df[self.df[protected_class_column] == cls][target_col].mean()
            ans += f'Среднее значение таргета по классу {cls} = {model_val}\n'
            model_vals.append(model_val)
            true_vals.append(self.df[self.df[protected_class_column] == cls][true_target_col].mean())
        subcategorybar(cats, [true_vals, model_vals], ['Факт', 'Прогноз'])
        return ans


model = CatBoostRegressor().load_model('/Users/m1crozavr/PycharmProjects/interpretability/data/model/model.cbm')
explainer = Explainer(model, df)

if __name__ == '__main__':
    model = CatBoostRegressor().load_model('/Users/m1crozavr/PycharmProjects/interpretability/data/model/model.cbm')
    exp = Explainer(model, df)
    print(f'{exp.get_shap_values(4)["Credit amount"].values:+.4f}')
