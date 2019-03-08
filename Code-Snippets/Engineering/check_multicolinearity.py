#Colinearity is the state where two variables are highly correlated and contain similiar information about the variance within a given dataset.
#To detect colinearity among variables, we simply create a correlation matrix
import seaborn as sns

#Make sure you work only on features X
sns.heatmap(data=X.corr(), annot=True).set_title('variables correlations')

#To choose what features we should remove, we'll use Variance Inflation Factor (VIF).
#We will onky keep all features if VIF < 5. If not, we'll remove the feature with the highest VIF and then run the snippet again.

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns 
vif.round(1)