imdb_data = read.csv("~/Desktop/IMDB_data_Fall_2024.csv")
attach(imdb_data)

set.seed(42)
library(dplyr)
library(caret)
library(ggplot2)
require(methods)
library(tm)
library(stringr)
library(car)
library(glmnet)
library(ggpubr)
library(ggplot2)
require(caTools)
require(splines)
library(splines) 
require(lmtest)
require(plm)



#Data frame consisting on only the relevant predictors (will update this as we move ahead)
imdbData1 = subset(imdb_data, select = -c(movie_title, movie_id, imdb_link))
attach(imdbData1)

imdbData1 = subset(imdbData1, select = -c(action, adventure, scifi, thriller, musical, romance, western, sport, horror, drama, war, animation, crime))

#Checking data types
str(imdbData1)


#Converting to factors
imdbData1$release_month = as.factor(release_month)
imdbData1$language = as.factor(language)
imdbData1$country = as.factor(country)
imdbData1$maturity_rating = as.factor(maturity_rating)
imdbData1$colour_film = as.factor(colour_film)


#Splitting the keywords
imdbData1$genres_split <- strsplit(as.character(genres), "\\|")
attach(imdbData1)

#Recombining the keywords for vector
imdbData1$genres_combined <- sapply(genres_split, function(x) paste(unlist(x), collapse = " "))
attach(imdbData1)

#Vector Matrix of the keywords
corpus = Corpus(VectorSource(genres_combined))

dtm = DocumentTermMatrix(corpus)

keywords = as.data.frame(as.matrix(dtm))

#Joining to the original data
imdbData1 = cbind(imdbData1, keywords)
attach(imdbData1)



#Final data set for regressions
finalData = subset(imdbData1, select = -c(distributor, director, actor1, actor2, actor3, genres, plot_keywords, cinematographer, production_company, genres_combined, genres_split)) 
attach(finalData)


#Create a new data frame with only numeric columns
numeric_columns = sapply(finalData, is.numeric)

finalData_numeric = subset(finalData, select = c(numeric_columns))

#Removing imdb score
finalData_numeric = subset(finalData_numeric, select = -c(imdb_score))

#Calculate and remove highly correlated numeric columns
cor_matrix = cor(finalData_numeric)
cor_matrix

highly_correlated = findCorrelation(cor_matrix, cutoff = 0.75, exact = TRUE)
highly_correlated

#Since no highly correlated variables were found, we will go ahead with our dataset

#Re-leveling the categorical variables


finalData$country <- relevel(finalData$country, ref = "USA")
finalData$release_month <- relevel(finalData$release_month, ref = "Jan")
finalData$language <- relevel(finalData$language, ref = "English")
finalData$maturity_rating <- relevel(finalData$maturity_rating, ref = "R")
finalData$isEnglish = ifelse(finalData$language == "English",1,0)



#linear regression with all variables
reg1 = lm(imdb_score ~., data = finalData)
summary(reg1)
vif(reg1)

#Error in vif means colleniearity. 
alias(reg1)
#Countries and languages are fully related to each other. We will be dropping countries because of
#having less factors and hence less overfitting
levels(country)
levels(language)

#Removing country

finalData1 = subset(finalData, select = -c(country))
attach(finalData1)

#Rerunning the regression
reg_lineartest = lm(imdb_score ~., data = finalData1)
summary(reg_lineartest)

vif(reg_lineartest)
####---- p-Values for Predictors in Linear Regression of IMDb Scores-----
# Extract coefficients and p-values for plotting significant variables
coef_names <- names(coef(reg_lineartest))
p_values <- summary(reg_lineartest)$coefficients[, 4]

# Remove language-related variables, release_month, and intercept from both the variable names and p-values
keep_indices <- !(coef_names %in% c("(Intercept)") | grepl("language", coef_names) | grepl("release_month", coef_names))
coef_names_filtered <- coef_names[keep_indices]
p_values_filtered <- p_values[keep_indices]

# Remove any NA values from the filtered variables and p-values
valid_indices <- !is.na(coef_names_filtered) & !is.na(p_values_filtered)
coef_names_filtered <- coef_names_filtered[valid_indices]
p_values_filtered <- p_values_filtered[valid_indices]

# Ensure the lengths of filtered coefficients and p-values match
if (length(coef_names_filtered) != length(p_values_filtered)) {
  stop("Mismatch between the number of coefficient names and p-values after filtering.")
}

# Create the data frame for p-values of the filtered variables
p_values_df <- data.frame(
  Variables = coef_names_filtered,  
  p_value = p_values_filtered       
)

# Add a column to determine significance (p-value < 0.05)
p_values_df$Significant <- p_values_df$p_value < 0.05

library(ggplot2)

# Plot -log10(p-values) to highlight significant variables
ggplot(p_values_df, aes(x = reorder(Variables, -log10(p_value)), y = -log10(p_value), fill = Significant)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(
    title = "p-Values from Linear Regression for Variable Significance",
    x = "Variables",
    y = "-log10(p-value)"
  ) +
  scale_fill_manual(values = c("FALSE" = "gray", "TRUE" = "cyan")) +  
  theme_minimal() +
  theme(
    axis.text = element_text(size = 6),         
    axis.title = element_text(size = 10),        
    plot.title = element_text(size = 12, hjust = 0.5)  
  )
df1 = finalData1


####################################################

####                    OUTLIERS           ########

####################################################

#BUDGET
#OUTLIER
o_b = lm(imdb_score~movie_budget, data=df1)
outlierTest(o_b)
qqPlot(o_b, envelope=list(style="none")) #316 & 989
nrow(df1)
df1[989, ]
df1[316,]
df1 = df1[-c(316, 989), ] #remove
nrow(df1)
rownames(df1) = NULL
o_b = lm(imdb_score~movie_budget, data=df1)
outlierTest(o_b)
df1 = df1[-c(1579), ]
rownames(df1) <- NULL
o_b = lm(imdb_score~movie_budget, data=df1)
outlierTest(o_b)


#duration
o_d = lm(imdb_score~duration, data=df1)
outlierTest(o_d) #3 outliers reported at 395 , 191, 1806
df1 = df1[-c(394,191,1803), ]
rownames(df1) <- NULL
o_d2 =lm(imdb_score~duration, data=df1)
qqPlot(o_d, envelope=list(style="none"))
outlierTest(o_d)
df1 = df1[-c(394,191,1803),]
rownames(df1) <- NULL
o_d3 =lm(imdb_score~duration, data=df1)
outlierTest(o_d3) # no outliers



#nb_news_articles
o3 = lm(imdb_score~nb_news_articles, data=df1)
outlierTest(o3) #487
df1 = df1[-c(487),]
rownames(df1) <- NULL
o3 = lm(imdb_score~nb_news_articles, data=df1)
outlierTest(o3_t)
df1 = df1[-c(492,1581),]
rownames(df1) <- NULL
o3_t = lm(imdb_score~nb_news_articles, data=df1)
outlierTest(o3_t)
summary(o3_t)
df1 = df1[-c(12),]
rownames(df1) <- NULL
o3_t = lm(imdb_score~nb_news_articles, data=df1)
outlierTest(o3_t) # all removed



#actor1_star_meter
o4 = lm(imdb_score~actor1_star_meter, data=df1)
outlierTest(o4)
#p  = 0.139 - ignore


#actor2_star_meter
o5 = lm(imdb_score~actor2_star_meter, data=df1)
outlierTest(o5)
#no outliers

#actor3_star_meter
o6 = lm(imdb_score~actor3_star_meter, data=df1)
outlierTest(o6)
#low p (0.05129) - ignore

#nb_faces
o7 = lm(imdb_score~nb_faces, data=df_test)
outlierTest(o7) #none



#movie_meter_IMDBpro
o8 = lm(imdb_score~movie_meter_IMDBpro, data=df1)
outlierTest(o8) 
#none

#reported later on - in imdb pro so removing now
df1=df1[-c(1246,1034), ]
rownames(df1) <- NULL


#####HETEROSKEDASTICITY  - CAN IGNORE THIS BIT FOR NOW

#budget
residualPlot(o_b, quadratic=FALSE) #no funnel
ncvTest(o_b)
#Chisquare = 31.85513, Df = 1, p = 1.6866e-08 (low p)

#duration 
residualPlot(o_d, quadratic=FALSE) #FUNNEL SHAPE REPORTED
ncvTest(o_d) # Chisquare = 25.54976, Df = 1, p = 4.3112e-07

df2 = df1

####################################################

####           POLY FITS REGRESSIONS        ########

####################################################


###################################################
#############BUDGET################################

#chosen - linear
#p 0.2090 for degree 2 so we choose a linear model
reg1_budget = lm(imdb_score ~ movie_budget, data = df2)
reg2_budget = lm(imdb_score ~ poly(movie_budget, 2), data = df2)
reg3_budget = lm(imdb_score ~ poly(movie_budget, 3), data = df2)
reg4_budget = lm(imdb_score ~ poly(movie_budget, 4), data = df2)

# Summary of each regression model
summary(reg1_budget) #r2 0.005227
summary(reg2_budget) #r2 0.006079
summary(reg3_budget) #r2 0.006082
summary(reg4_budget) #r2 0.007276

anova(reg1_budget, reg2_budget, reg3_budget, reg4_budget)


###PLOT 
#plot enviornment
plot_budget = ggplot(df2, aes(x = movie_budget, y = imdb_score)) +
  geom_point(color = "darkgray", alpha = 0.6, size = 1.5) +
  geom_smooth(method = "lm", formula = y ~ x, aes(color = "Degree 1"), se = FALSE) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), aes(color = "Degree 2"), se = FALSE) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), aes(color = "Degree 3"), se = FALSE) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 4), aes(color = "Degree 4"), se = FALSE) +
  scale_color_manual(name = "Polynomial Degree", 
                     values = c("Degree 1" = "cyan3", 
                                "Degree 2" = "darkgoldenrod1", 
                                "Degree 3" = "cornflowerblue", 
                                "Degree 4" = "darkorchid2")) +
  labs(title = "Relationship Between Movie Budget and IMDb Score",
       x = "Movie Budget (in USD)",
       y = "IMDb Score",
       color = "Polynomial Degree") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

print(plot_budget)

###CHECKING MODEL
#checking for model issues, outliers checked for earlier
residualPlot(reg1_budget) #looks linear
#heteroskedasticity
residualPlot(reg1_budget, quadratic=FALSE) # dont see a funnel shape
ncvTest(reg1_budget)#31.82553, Df = 1, p = 1.6866e-08
#modify to fix for heteroskedasticity -- note - will actually have to fix final model
reg1_budget_2 = coeftest(reg1_budget, vcov=vcovHC(reg1_budget, type="HC1"))
reg1_budget_2 #p value 0.0004782, vs earlier p value of 0.000761 ***
#outlier test
outlierTest(reg1_budget)



######DURATION#####
#chosen - SPLINE , 3 knots, d=1
#alternate choice: poly3
reg1_duration = lm(imdb_score ~ duration, data = df2)
reg2_duration = lm(imdb_score ~ poly(duration, 2), data = df2)
reg3_duration = lm(imdb_score ~ poly(duration, 3), data = df2)
reg4_duration = lm(imdb_score ~ poly(duration, 4), data = df2)
reg5_duration = lm(imdb_score~poly(duration,5), data=df2)


#plot enviornment
plot_duration = ggplot(df2, aes(x = duration, y = imdb_score)) +
  geom_point(color = "darkgray", alpha = 0.6, size = 1.5) +
  geom_smooth(method = "lm", formula = y ~ x, aes(color = "Degree 1"), se = FALSE) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), aes(color = "Degree 2"), se = FALSE) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), aes(color = "Degree 3"), se = FALSE) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 4), aes(color = "Degree 4"), se = FALSE) +
  scale_color_manual(name = "Polynomial Degree", 
                     values = c("Degree 1" = "cyan3", 
                                "Degree 2" = "darkgoldenrod1", 
                                "Degree 3" = "cornflowerblue", 
                                "Degree 4" = "darkorchid2")) +
  labs(title = "Relationship Between Movie Duration and IMDb Score with Polynomial Fits",
       x = "Duration (in Minutes)",
       y = "IMDb Score",
       color = "Polynomial Degree") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

print(plot_duration)

cplot_duration2 = plot_duration + scatter_plot + l1_duration + l2_duration + l3_duration + l4_duration + 
  scale_color_manual(name = "Polynomial Degree", values = c("Degree 1" = "cyan3", "Degree 2" = "darkgoldenrod1", "Degree 3" = "cornflowerblue", "Degree 4" = "darkorchid"))


# Summary of each regression model
summary(reg1_duration) # r^2 0.1748,	Adjusted R-squared:  0.1744
summary(reg2_duration) #0.1945,	Adjusted R-squared:  0.1933 p-value: < 2.2e-16
summary(reg3_duration) #r^2 0.1948
summary(reg4_duration) #0.2025
summary(reg5_duration) #0.2082

anova(reg1_duration, reg2_duration, reg3_duration, reg4_duration)
# reg3_duration is worse with p=0.4462 and reg4 p = 2.87e^-5, so discarding reg3

anova(reg1_duration, reg2_duration, reg4_duration, reg5_duration)
#added one more regression to compare, didnt add much more so we will go with cubic if we choose polynomial

#try different k

k1_d = quantile(duration, 0.05)
k2_d = quantile(duration,0.5)
k3_d=quantile(duration,0.95)
#k4_d=quantile(duration,0.95)


#splines
spline1_duration=lm(imdb_score~ bs(duration,knots=c(k1_d,k2_d,k3_d),degree=1), data = df2)
spline2_duration=lm(imdb_score ~ bs(duration,knots=c(k1_d,k2_d,k3_d),degree=2), data = df2)
spline3_duration=lm(imdb_score ~ bs(duration,knots=c(k1_d,k2_d,k3_d),degree=3), data = df2)         
spline4_duration=lm(imdb_score ~ bs(duration,knots=c(k1_d,k2_d,k3_d),degree=4), data = df2)
spline5_duration=lm(imdb_score ~ bs(duration,knots=c(k1_d,k2_d,k3_d),degree=5), data = df2)

#spline lines for graph
spline_l1_dur= geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_d,k2_d,k3_d)), color = "cyan3")
spline_l2_dur = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_d,k2_d,k3_d), degree = 2), color = "darkgoldenrod1")
spline_l3_dur = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_d,k2_d,k3_d), degree = 3), color = "cornflowerblue")
spline_l4_dur = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_d,k2_d,k3_d), degree = 4), color = "darkorchid2")
spline_l5_dur = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_d,k2_d,k3_d), degree = 5), color = "coral3")

#lines at knot
klines = geom_vline(xintercept = c(k1_d,k2_d,k3_d), linetype = "dashed", color = "black")


#plot
plot_duration + scatter_plot + spline_l1_dur + spline_l2_dur + spline_l3_dur + spline_l4_dur + spline_l5_dur + klines

summary(spline1_duration) #  0.2099,p-value: < 2.2e-16
summary(spline2_duration) #0.2104 p-value: < 2.2e-16    
summary(spline3_duration) #R^2: 0.2111
summary(spline4_duration) #R^2 0.2117
summary(spline5_duration) #R^2 0.2115
anova(spline1_duration,spline3_duration,spline4_duration)
#spline2 goes down, check without, p is 0.29. so stick to linear model.


#testing 
outlierTest(spline1_duration) #no outliers
ncvTest(spline1_duration) #< 2.22e-16
spline3_duration_adj = coeftest(spline3_duration, vcov=vcovHC(spline3_duration, type="HC1"))
spline3_duration_adj 
  



###RELEASE YEAR
#chosen - linear spline
#PROBABLY DON'T ACTUALLY NEED IT GIVEN ALL OUR MOVIES ARE IN THIS YEAR?
# BUT MAY CHANGE OVERALL PREDICTIVE POWER OF THE MODEL?

#chosen - 
reg1_yr = lm(imdb_score ~ release_year, data = df2)
reg2_yr = lm(imdb_score ~ poly(release_year, 2), data = df2)
reg3_yr = lm(imdb_score ~ poly(release_year, 3), data = df2)
reg4_yr = lm(imdb_score ~ poly(release_year, 4), data = df2)
residualPlot(reg3_yr)

#plot enviornment
plot_yr = ggplot(df2, aes(x = release_year, y = imdb_score)) +
  geom_point(color = "gray", size = 1.5, alpha = 0.6) +
  geom_smooth(method = "lm", formula = y ~ x, aes(color = "Linear (Degree 1)"), se = FALSE, size = 1.2, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), aes(color = "Polynomial (Degree 2)"), se = FALSE, size = 1.2, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), aes(color = "Polynomial (Degree 3)"), se = FALSE, size = 1.2, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 4), aes(color = "Polynomial (Degree 4)"), se = FALSE, size = 1.2, linetype = "solid") +
  labs(title = "Relationship Between Release Year and IMDb Score with Polynomial Fits",
       x = "Release Year",
       y = "IMDb Score",
       color = "Model Type") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "bottom",
        axis.title = element_text(face = "plain")) +
  scale_color_manual(name = "Polynomial Degree",
                     values = c("Linear (Degree 1)" = "cyan3", 
                                "Polynomial (Degree 2)" = "darkgoldenrod1", 
                                "Polynomial (Degree 3)" = "cornflowerblue", 
                                "Polynomial (Degree 4)" = "darkorchid2"))

# Display the plot
plot_yr

# Summary of each regression model
summary(reg1_yr) #0.03743 p-value: < 2.2e-16
summary(reg2_yr) #0.04224
summary(reg3_yr) #0.04632
summary(reg4_yr) #0.0469

anova(reg1_yr, reg2_yr, reg3_yr, reg4_yr)
#### given p of 0.0009367 , choosing deg=3

k1_y= 1955
k2_y = 2005
k3_y = 2005

#tried increasing k to 4 knots but that didnt change much. 0.1858 instad of 0.1851 r2 for spline1 instead of 
#evaluating moving splines not baed on percentile but visually?


#splines
spline1_yr=lm(imdb_score~ bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1),data=df2)
spline2_yr=lm(imdb_score ~ bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=2),data=df2)
spline3_yr=lm(imdb_score ~ bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=3),data=df2)         
spline4_yr=lm(imdb_score ~ bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=4),data=df2)
spline5_yr=lm(imdb_score ~ bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=5),data=df2)

#spline lines for graph
spline_l1_yr= geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_y, k2_y, k3_y)), color = "cyan3")
spline_l2_yr = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_y, k2_y, k3_y), degree = 2), color = "darkgoldenrod1")
spline_l3_yr = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_y, k2_y, k3_y), degree = 3), color = "cornflowerblue")
spline_l4_yr = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_y, k2_y, k3_y), degree = 4), color = "darkorchid2")
spline_l5_yr = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_y, k2_y, k3_y), degree = 5), color = "coral3")

#lines at knot
klines_yr = geom_vline(xintercept = c(k1_y, k2_y, k3_y), linetype = "dashed", color = "black")


#plot
plot_yr + scatter_plot + spline_l1_yr + spline_l2_yr + spline_l3_yr + spline_l4_yr + spline_l5_yr + klines_yr

summary(spline1_yr) #0.04753 -- similar to cubic but can choose a simpler method
summary(spline2_yr) #r2 = 0.04632
summary(spline3_yr) #r2 =0.04726
summary(spline4_yr) #r2 =0.04736 #remove from comparison
summary(spline5_yr) #r2=0.04756
anova(spline1_yr,spline3_yr) #0.02702 p - but may not be worth going to such a high degree spline. increase k??
##spline1 chosen


#testing
residualPlot(spline1_yr)
ncvTest(spline1_yr) #p = 0.3 - ok now

outlierTest(spline1_yr)
#no outliers



################NUMBER OF NEWS ARTICLES
#chosen - 
###CHOOSING SPLINE D=3, BUT CAN CHOOSE CUBIC (polynomial for simplification)

reg1_articles = lm(imdb_score ~ nb_news_articles, data = df2)
reg2_articles = lm(imdb_score ~ poly(nb_news_articles, 2), data = df2)
reg3_articles = lm(imdb_score ~ poly(nb_news_articles, 3), data = df2)
reg4_articles = lm(imdb_score ~ poly(nb_news_articles, 4), data = df2)
outlierTest(reg1_articles)


#plot enviornment
cplot_articles = ggplot(df2, aes(x = nb_news_articles, y = imdb_score)) +
  geom_point(color = "gray", size = 1.5, alpha = 0.6) +
  geom_smooth(method = "lm", formula = y ~ x, aes(color = "Linear (Degree 1)"), se = FALSE, size = 1.2, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), aes(color = "Polynomial (Degree 2)"), se = FALSE, size = 1.2, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), aes(color = "Polynomial (Degree 3)"), se = FALSE, size = 1.2, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 4), aes(color = "Polynomial (Degree 4)"), se = FALSE, size = 1.2, linetype = "solid") +
  labs(title = "Relationship Between Number of News Articles and IMDb Score with Polynomial Fits",
       x = "Number of News Articles",
       y = "IMDb Score",
       color = "Model Type") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "bottom",
        legend.title = element_text(face = "bold"),
        axis.title = element_text(face = "plain")) +
  scale_color_manual(name = "Polynomial Degree",
                     values = c("Linear (Degree 1)" = "cyan3", 
                                "Polynomial (Degree 2)" = "darkgoldenrod1", 
                                "Polynomial (Degree 3)" = "cornflowerblue", 
                                "Polynomial (Degree 4)" = "darkorchid2"))

# Display the plot
cplot_articles

# Summary of each regression model

summary(reg1_articles)#0.09665
summary(reg2_articles) #0.121
summary(reg3_articles) #0.1307
summary(reg4_articles) #0.1344
outlierTest(reg3_articles)

anova(reg1_articles, reg2_articles, reg3_articles, reg4_articles)
#p for d2 = 5.515e-13, d3 = 6.337e-06, d4 = 0.006389 -- given we will be getting more complex stop at cubic. evaluate splines or log


k1_a = 2000
k2_a = 6000
k3_a = 6000

#splines
spline1_articles=lm(imdb_score~ bs(nb_news_articles,knots=c(k1_a,k2_a,k3_a),degree=1),data=df2)
spline2_articles=lm(imdb_score ~ bs(nb_news_articles,knots=c(k1_a,k2_a,k3_a),degree=2), data=df2)
spline3_articles=lm(imdb_score ~ bs(nb_news_articles,knots=c(k1_a,k2_a,k3_a),degree=3),data=df2)         
spline4_articles=lm(imdb_score ~ bs(nb_news_articles,knots=c(k1_a,k2_a,k3_a),degree=4),data=df2)
spline5_articles=lm(imdb_score ~ bs(nb_news_articles,knots=c(k1_a,k2_a,k3_a),degree=5),data=df2)

#lines at knot
klines_art = geom_vline(xintercept = c(k1_a,k2_a,k3_a), linetype = "dashed", color = "black")

#plot
plot_articles <- ggplot(df2, aes(x = nb_news_articles, y = imdb_score)) +
  geom_point(color = "gray", size = 2, alpha = 0.5) +
  spline_l1_articles +
  spline_l2_articles +
  spline_l3_articles +
  spline_l4_articles +
  spline_l5_articles +
  klines_art +
  labs(title = "Relationship Between Number of News Articles and IMDb Score with Spline Fits",
       x = "Number of News Articles",  # Updated to use full spelling
       y = "IMDb Score") +
  theme_classic()

# Display the updated plot
plot_articles

##knots at 2000 & 6000, 1000 
#d1 - 0.1269, d2-0.1317, d3=0.1334
##knots at 2000 & 6000:
#d1 =0.1269
#d2=0.1316
#d3=0.1335

#earlier quantile based
#r2 = 0.1049 (quantile based)
#r2 = 0.1241 -- 
#r2 = 0.1227 -- goes down
##0.1398
outlierTest(spline3_articles)


######NB_FACES
#chosen - #LINEAR MODEL CHOSEN - overall seems to be low impact? 
reg1_faces = lm(imdb_score ~ nb_faces, data = df2)
reg2_faces = lm(imdb_score ~ poly(nb_faces, 2), data = df2)
reg3_faces = lm(imdb_score ~ poly(nb_faces, 3), data = df2)
reg4_faces = lm(imdb_score ~ poly(nb_faces, 4), data = df2)



#plot enviornment
plot_faces = ggplot(df2, aes(x = nb_faces, y = imdb_score)) +
  geom_point(color = "gray", size = 1.5, alpha = 0.6) +
  geom_smooth(method = "lm", formula = y ~ x, aes(color = "Degree 1"), se = FALSE, size = 1.2, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), aes(color = "Degree 2"), se = FALSE, size = 1.2, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), aes(color = "Degree 3"), se = FALSE, size = 1.2, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 4), aes(color = "Degree 4"), se = FALSE, size = 1.2, linetype = "solid") +
  labs(title = "Relationship Between Number of Faces in the Main Movie Poster and IMDb Score with Polynomial Fits",
       x = "Number of Faces in the Main Movie Poster",
       y = "IMDb Score") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "bottom",
        legend.title = element_text(face = "bold"),
        axis.title = element_text(face = "bold")) +
  scale_color_manual(name = "Polynomial Degree",
                     values = c("Degree 1" = "cyan3", "Degree 2" = "darkgoldenrod1", "Degree 3" = "cornflowerblue", "Degree 4" = "darkorchid2"))

plot_faces

# Summary of each regression model
summary(reg1_faces) #0.0068
summary(reg2_faces) #0.00866
summary(reg3_faces) #0.008732
summary(reg4_faces) #0.008747

anova(reg1_faces, reg2_faces, reg3_faces, reg4_faces)
#### given p for d2 = 0.07 so we use linear 

#spline
k1_faces= 10
k2_faces= 10
k3_faces = 10


#splines
spline1_faces=lm(imdb_score~ bs(nb_faces,knots=c(k1_faces, k2_faces, k3_faces),degree=1),data=df2)
spline2_faces=lm(imdb_score ~ bs(nb_faces,knots=c(k1_faces, k2_faces, k3_faces),degree=2),data=df2)
spline3_faces=lm(imdb_score ~ bs(nb_faces,knots=c(k1_faces, k2_faces, k3_faces),degree=3),data=df2)         
spline4_faces=lm(imdb_score ~ bs(nb_faces,knots=c(k1_faces, k2_faces, k3_faces),degree=4),data=df2)
spline5_faces=lm(imdb_score ~ bs(nb_faces,knots=c(k1_faces, k2_faces, k3_faces),degree=5),data=df2)

#spline lines for graph
spline_l1_faces= geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_faces, k2_faces, k3_faces)), color = "cyan3")
spline_l2_faces = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_faces, k2_faces, k3_faces), degree = 2), color = "darkgoldenrod1")
spline_l3_faces = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_faces, k2_faces, k3_faces), degree = 3), color = "cornflowerblue")
spline_l4_faces = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_faces, k2_faces, k3_faces), degree = 4), color = "darkorchid2")
spline_l5_faces = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_faces, k2_faces, k3_faces), degree = 5), color = "coral3")

#lines at knot
klines_faces = geom_vline(xintercept = c(k1_faces, k2_faces, k3_faces), linetype = "dashed", color = "black")


#plot
plot_faces + scatter_plot + spline_l1_faces + spline_l2_faces + spline_l3_faces + spline_l4_faces + spline_l5_faces + klines_faces

summary(spline1_faces)  #0.008838 p-value: 0.0007175
summary(spline2_faces) #0.008842 p-value: 0.00456
summary(spline3_faces) #0.009489 p-value: 0.005729
summary(spline5_faces) #0.01158 p-value 0.004494
anova(spline1_faces,spline2_faces, spline3_faces) 


####STAR1 METER
#chosen - 
reg1_star1 = lm(imdb_score ~ actor1_star_meter, data = df2)
reg2_star1 = lm(imdb_score ~ poly(actor1_star_meter, 2), data = df2)
reg3_star1 = lm(imdb_score ~ poly(actor1_star_meter, 3), data = df2)
reg4_star1 = lm(imdb_score ~ poly(actor1_star_meter, 4), data = df2)




#plot enviornment
plot_star1 = ggplot(df2, aes(x = actor1_star_meter, y = imdb_score)) +
  geom_point(color = "gray", size = 2, alpha = 0.5) + # Increase point size and set transparency for better visual distinction
  geom_smooth(method = "lm", formula = y ~ x, aes(color = "Degree 1"), se = FALSE, size = 1.5, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), aes(color = "Degree 2"), se = FALSE, size = 1.5, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), aes(color = "Degree 3"), se = FALSE, size = 1.5, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 4), aes(color = "Degree 4"), se = FALSE, size = 1.5, linetype = "solid") +
  labs(title = "Relationship Between Actor 1 Star Meter vs IMDb Score with Polynomial Fits",
       x = "Actor 1 Star Meter (Lower Value = More Famous)",
       y = "IMDb Score") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "bottom",
        legend.title = element_text(face = "bold"),
        axis.title = element_text(face = "bold"),
        axis.text = element_text(size = 10)) +
  scale_color_manual(name = "Polynomial Degree",
                     values = c("Degree 1" = "cyan3", 
                                "Degree 2" = "darkgoldenrod1", 
                                "Degree 3" = "cornflowerblue", 
                                "Degree 4" = "darkorchid2")) 

# Display the updated plot
cplot_star1 = plot_star1
cplot_star1

# Summary of each regression model
summary(reg1_star1) #0.0008514
summary(reg2_star1) #0.001346
summary(reg3_star1) #0.001401
summary(reg4_star1) #0.002921

anova(reg1_star1, reg2_star1, reg3_star1, reg4_star1)
#### given p of 0.3418 at d1, Just use linear model/ ignore

k1_star1= 1e6
k2_star1=4e6
k3_star1 = 1/2


#splines
spline1_star1=lm(imdb_score~ bs(actor1_star_meter,knots=c(k1_star1, k2_star1, k3_star1),degree=1),data=df2)
spline2_star1=lm(imdb_score ~ bs(actor1_star_meter,knots=c(k1_star1, k2_star1, k3_star1),degree=2),data=df2)
spline3_star1=lm(imdb_score ~ bs(actor1_star_meter,knots=c(k1_star1, k2_star1, k3_star1),degree=3),data=df2)         
spline4_star1=lm(imdb_score ~ bs(actor1_star_meter,knots=c(k1_star1, k2_star1, k3_star1),degree=4),data=df2)
spline5_star1=lm(imdb_score ~ bs(actor1_star_meter,knots=c(k1_star1, k2_star1, k3_star1),degree=5),data=df2)

#spline lines for graph
spline_l1_star1= geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_star1, k2_star1, k3_star1)), color = "cyan3")
spline_l2_star1 = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_star1, k2_star1, k3_star1), degree = 2), color = "darkgoldenrod1")
spline_l3_star1 = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_star1, k2_star1, k3_star1), degree = 3), color = "cornflowerblue")
spline_l4_star1 = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_star1, k2_star1, k3_star1), degree = 4), color = "darkorchid2")
spline_l5_star1 = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_star1, k2_star1, k3_star1), degree = 5), color = "coral3")

#lines at knot
klines_star1 = geom_vline(xintercept = c(k1_star1, k2_star1, k3_star1), linetype = "dashed", color = "black")


#plot
plot_star1 + scatter_plot + spline_l1_star1 + spline_l2_star1 + spline_l3_star1 + spline_l4_star1 + spline_l5_star1 + klines_star1

summary(spline1_star1) # 0.001726
summary(spline2_star1) # 0.001967
summary(spline3_star1) # 0.003937
summary(spline4_star1) # 0.008145
summary(spline5_star1) # 0.01047
anova(spline1_star1,spline5_star1) #

#placing at 1e6 and 2 e6
#R2 0.00336 0.1687
#deg 2 0.003031 p-value: 0.3256

#placing at 1e6 and 2 e6 + 0.5 of the data
#R2 0.00197   p p-value: 0.3449
#deg 2 0.003031 p-value:0.4378

###IMDBPRO
#chosen - spline2_pro

reg1_pro = lm(imdb_score ~ movie_meter_IMDBpro, data = df2)
reg2_pro = lm(imdb_score ~ poly(movie_meter_IMDBpro, 2), data = df2)
reg3_pro = lm(imdb_score ~ poly(movie_meter_IMDBpro, 3), data = df2)
reg4_pro = lm(imdb_score ~ poly(movie_meter_IMDBpro, 4), data = df2)
reg5_pro = lm(imdb_score ~ poly(movie_meter_IMDBpro, 5), data = df2)




#plot enviornment
plot_pro = ggplot(df2, aes(x = movie_meter_IMDBpro, y = imdb_score)) +
  geom_point(color = "gray", size = 2, alpha = 0.5) + # Increase point size and set transparency for better visibility
  geom_smooth(method = "lm", formula = y ~ x, aes(color = "Degree 1"), se = FALSE, size = 1.5, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), aes(color = "Degree 2"), se = FALSE, size = 1.5, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), aes(color = "Degree 3"), se = FALSE, size = 1.5, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 4), aes(color = "Degree 4"), se = FALSE, size = 1.5, linetype = "solid") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 5), aes(color = "Degree 5"), se = FALSE, size = 1.5, linetype = "solid") +
  labs(title = "Relationship Between IMDbPro Movie Meter vs IMDb Score with Polynomial Fits",
       x = "IMDbPro Movie Meter (Lower Value = More Popular)",
       y = "IMDb Score") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "bottom",
        legend.title = element_text(face = "bold"),
        axis.title = element_text(face = "bold"),
        axis.text = element_text(size = 10)) +
  scale_color_manual(name = "Polynomial Degree",
                     values = c("Degree 1" = "cyan3", 
                                "Degree 2" = "darkgoldenrod1", 
                                "Degree 3" = "cornflowerblue", 
                                "Degree 4" = "darkorchid2",
                                "Degree 5" = "coral3")) 

# Display the updated plot
cplot_pro = plot_pro
cplot_pro

# Summary of each regression model
summary(reg1_pro) #r2 0.008736
summary(reg2_pro) # 0.04172
summary(reg3_pro) # 0.07405
summary(reg4_pro) # 0.0903
summary(reg5_pro) #r2 0.1268

anova(reg1_pro, reg2_pro, reg3_pro, reg4_pro,reg5_pro)
#### each model performs significantly better than previous. evaluate splines


k1_p= quantile(movie_meter_IMDBpro, 0.5)
k2_p = quantile(movie_meter_IMDBpro, 0.99)
k3_p = quantile(movie_meter_IMDBpro, 0.99)
#tried increasing k to 4 knots but that didnt change much. 0.1858 instad of 0.1851 r2 for spline1 instead of 
#evaluating moving splines not bad on percentile but visually?


#splines
spline1_pro=lm(imdb_score~ bs(movie_meter_IMDBpro,knots=c(k1_p, k2_p, k3_p),degree=1),data=df2)
spline2_pro=lm(imdb_score ~ bs(movie_meter_IMDBpro,knots=c(k1_p, k2_p, k3_p),degree=2),data=df2)
spline3_pro=lm(imdb_score ~ bs(movie_meter_IMDBpro,knots=c(k1_p, k2_p, k3_p),degree=3),data=df2)         
spline4_pro=lm(imdb_score ~ bs(movie_meter_IMDBpro,knots=c(k1_p, k2_p, k3_p),degree=4),data=df2)
spline5_pro=lm(imdb_score ~ bs(movie_meter_IMDBpro,knots=c(k1_p, k2_p, k3_p),degree=5),data=df2)

#spline lines for graph
spline_l1_pro= geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_p, k2_p, k3_p)), color = "cyan3")
spline_l2_pro = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_p, k2_p, k3_p), degree = 2), color = "darkgoldenrod2")
spline_l3_pro = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_p, k2_p, k3_p), degree = 3), color = "cornflowerblue")
spline_l4_pro = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_p, k2_p, k3_p), degree = 4), color = "darkorchid2")
spline_l5_pro = geom_smooth(method = "lm", formula = y ~ bs(x, knots = c(k1_p, k2_p, k3_p), degree = 5), color = "coral3")

#lines at knot
klines_pro = geom_vline(xintercept = c(k1_p, k2_p, k3_p), linetype = "dashed", color = "black")


#plot
plot_pro + scatter_plot + spline_l1_pro + spline_l2_pro + spline_l3_pro + spline_l4_pro + spline_l5_pro + klines_pro

summary(spline1_pro) #r2=0.1884 
summary(spline2_pro) #r2 = 0.1955  # improves
summary(spline3_pro) #r2 =0.1927 # lower than earlier
summary(spline4_pro) #r2 =0.1927 
summary(spline5_pro) #r2 =0.1909
anova(spline1_pro,spline2_pro) #05.534e-05 

outlierTest(spline2_pro)
#2 reported with p<0.05 at 1246 & 1034, remove in earlier part where outliers are


#2 knots, 1 at 0.5 and one at 0.99
#R2 values at increasing degrees 0.18, 0.1872, 0.1885

#3 knots - added one at 0.01
#R2 = 0.1819, 0.1874,0.1895 
# very close so just keep 2


#######################################################
#####CHOSEN NUMERIC MODELS#############################


#duration
summary(spline1_duration) #R-squared:  0.2099 p-value: < 2.2e-16
 

#imdbpro rating
summary(spline2_pro) #R2=0.1955 p-value: < 2.2e-16


## of articles
summary(spline3_articles) #R2 0.1362  p-value: < 2.2e-16

#release year
summary(spline1_yr) #0.04753 p-value: < 2.2e-16 #simpler model?

#budget
summary(reg1_budget) #R-squared:  0.005227 p-value: 0.0007606

#no. of faces 
summary(spline1_faces)  #R-squared: 0.008838 #p-value: 0.0007175


#star1 rating
summary(reg1_star1) #R-squared:  0.0008514 p-value: 0.2018


###############################################
####top categorical options###################
reg_drama = lm(imdb_score~drama, data=df2)
summary(reg_drama) #R-squared:  0.1152,	Adjusted R-squared:  0.1148 

reg_biography= lm(imdb_score~biography, data=df2)
summary(reg_biography) # R-squared:  0.03827,	Adjusted R-squared:  0.03777 

reg_comedy = lm(imdb_score~comedy, data=df2)
summary(reg_comedy) #R-squared:    0.03498,	Adjusted R-squared:  0.03448 


reg_horror = lm(imdb_score~horror, data=df2)
summary(reg_horror) #R-squared:  0.02872,	Adjusted R-squared:  0.02821 

 
reg_action = lm(imdb_score~action, data=df2)
summary(reg_action) #R-squared:    0.02748,	Adjusted R-squared:  0.02697 

reg_music = lm(imdb_score~music, data=df2)
summary(music) #R-squared:    0.02748,	Adjusted R-squared:  0.02697 

reg_western = lm(imdb_score~western,data=df2)
summary(reg_western) # 0.004315,	Adjusted R-squared:  0.003798 

reg_family = lm(imdb_score~family,data=df2)
summary(reg_family) #0.008064,	Adjusted R-squared:  0.007549 

reg_documentary= lm(imdb_score~documentary,data=df2)
summary(reg_documentary) # r2: 0.005979,	Adjusted R-squared:  0.005462 



reg_adventure = lm(imdb_score~adventure, data=df2)
summary(reg_adventure) #R-squared:    0.004886,	Adjusted R-squared:  0.004368

reg_crime = lm(imdb_score~crime, data=df2)
summary(reg_crime) #R-squared:   0.003588,	Adjusted R-squared:  0.00307 


reg_sport = lm(imdb_score~sport, data=df2)
summary(reg_sport) #R-squared:  0.002987,	Adjusted R-squared:  0.002469 


reg_thriller = lm(imdb_score~thriller, data=df2)
summary(reg_thriller) #R-squared:   0.007013,	Adjusted R-squared:  0.006497  


reg_animation = lm(imdb_score~animation, data=df2)
summary(reg_animation) #R-squared:  0.0002666,	Adjusted R-squared:  -0.000253 


reg_romance = lm(imdb_score~romance, data=df2)
summary(reg_romance) #R-squared:    0.0002364,	Adjusted R-squared:  -0.0002832




#importing test data

test = read.csv("~/Desktop/test_data_IMDB_Fall_2024.csv")
test1 = subset(test, select = -c(action, adventure, scifi, thriller, musical, romance, western, sport, horror, drama, war, animation, crime))
#All genres dummy variables
all_genres = c("action", "adventure", "scifi", "thriller", "musical", 
               "romance", "western", "sport", "horror", "drama", 
               "war", "animation", "crime", "biography", "comedy", 
               "music", "mystery", "fantasy", "family", "documentary")

#Splitting the keywords
test1$genres_split <- strsplit(as.character(test1$genres), "\\|")
#Recombining the keywords for vector
test1$genres_combined <- sapply(test1$genres_split, function(x) paste(unlist(x), collapse = " "))
#Vector Matrix of the keywords
corpus = Corpus(VectorSource(test1$genres_combined))
dtm = DocumentTermMatrix(corpus)
keywords = as.data.frame(as.matrix(dtm))

# Step 6: Ensure all original genre columns are present (Fill with 0s if missing)
# Add missing genre columns to the keywords data frame
for (genre in all_genres) {
  if (!(genre %in% colnames(keywords))) {
    keywords[[genre]] = 0  # Add the missing column and fill with 0s
  }
}
#Joining to the original data
testFinal = cbind(test1, keywords)


testFinal$release_month = as.factor(testFinal$release_month)
testFinal$language = as.factor(testFinal$language)
testFinal$maturity_rating = as.factor(testFinal$maturity_rating)
testFinal$colour_film = as.factor(testFinal$colour_film)
testFinal$isEnglish = ifelse(testFinal$language == "English",1,0)

#Releveling
testFinal$country <- relevel(testFinal$country, ref = "USA")
testFinal$release_month <- relevel(testFinal$release_month, ref = "Jan")
testFinal$language <- relevel(testFinal$language, ref = "English")
testFinal$maturity_rating <- relevel(testFinal$maturity_rating, ref = "R")




#####MULTIPLE LINEAR REGRESSION######

#duration and movie meter imdb pro
reg1 = lm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
           + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2),
           data = df2)
summary(reg1) #R-squared:  0.3214,	Adjusted R-squared:  0.3185 p-value: < 2.2e-16

reg1_cv = glm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
          + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2),
          data = df2)

cvreg1 = cv.glm(df2, reg1_cv)$delta[1]
cvreg1 #0.7751

#add number of news articles
reg2 <- lm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
           + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
           + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2),
           data = df2)
summary(reg2) #R-squared:  0.3364,	Adjusted R-squared:  0.3322 p-value: < 2.2e-16

reg2_cv <- glm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
           + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
           + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2),
           data = df2)

cvreg2 = cv.glm(df2, reg2_cv)$delta[1]
cvreg2 #0.760


#Adding release year
reg3 = lm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
          + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
          + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
            bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1),
           data = df2)
summary(reg3) #R-squared:  0.3622,	Adjusted R-squared:  0.3568 p-value: < 2.2e-16 

reg3_cv = glm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
          + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
          + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
            bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1),
          data = df2)

cvreg3 = cv.glm(df2, reg3_cv)$delta[1]
cvreg3 #0.7306


#Add budget
reg4 = lm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
          + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
          + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
            bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) +
           + movie_budget,
           data = df2)
summary(reg4) #R-squared:  0.3963,	Adjusted R-squared:  0.3908 p-value: < 2.2e-16 

reg4_cv = glm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
          + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
          + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
            bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) +
            + movie_budget,
          data = df2)

cvreg4 = cv.glm(df2, reg4_cv)$delta[1]
cvreg4 #0.6929


#Adding higher fsignificance genre variables
reg5 = lm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
          + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
          + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
            bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) +
            + movie_budget + drama + biography + comedy,
          data = df2)
summary(reg5) #R-squared:  0.4471,	Adjusted R-squared:  0.4412 p-value: < 2.2e-16 

reg5_cv = glm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
          + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
          + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
            bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) +
            + movie_budget + drama + biography + comedy,
          data = df2)

cvreg5 = cv.glm(df2, reg5_cv)$delta[1]
cvreg5 #0.6375

#Three additional genre variables
reg6 = lm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
          + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
          + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
            bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) +
            + movie_budget + drama + biography + comedy + horror + action + music,
          data = df2)
summary(reg6) #R-squared:  0.4738,	Adjusted R-squared:  0.4674 p-value: < 2.2e-16 

reg6_cv = glm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
          + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
          + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
            bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) +
            + movie_budget + drama + biography + comedy + horror + action + music,
          data = df2)

cvreg6 = cv.glm(df2, reg6_cv)$delta[1]
cvreg6 #0.6092


#Adding family and music to the model
reg7 = lm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
           + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
           + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
             bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) +
             + movie_budget + drama + biography + comedy + horror + action + music + family + music,
           data = df2)
summary(reg7) #R-squared:  0.4749,	Adjusted R-squared:  0.4681 p-value: < 2.2e-16 

reg7_cv = glm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
          + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
          + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
            bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) +
            + movie_budget + drama + biography + comedy + horror + action + music + family + music,
          data = df2)

cvreg7 = cv.glm(df2, reg7_cv)$delta[1]
cvreg7 #0.6092


#Adding romance and adventure
reg8 = lm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
            + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
            + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
              bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) 
            + movie_budget + drama + biography + crime + comedy + horror + action + family + music + romance + adventure,
            data = df2)
summary(reg8) #R-squared:  0.4789,	Adjusted R-squared:  0.4713 p-value: < 2.2e-16 

reg8_cv = glm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
          + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
          + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
            bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) 
          + movie_budget + drama + biography + crime + comedy + horror + action + family + music + romance + adventure,
          data = df2)

cvreg8 = cv.glm(df2, reg8_cv)$delta[1]
cvreg8 #0.606

#Adding animation and documentary
reg9 = lm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
          + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
          + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
            bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) 
          + movie_budget + drama + biography + crime + comedy + horror + action + family + music + romance + adventure + animation + documentary,
          data = df2)
summary(reg9) #R-squared:  0.4969,	Adjusted R-squared:  0.4891 p-value: < 2.2e-16 

reg9_cv = glm(imdb_score ~ bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
              + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
              + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
                bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) 
              + movie_budget + drama + biography + crime + comedy + horror + action + family + music + romance + adventure + animation + documentary,
              data = df2)

cvreg9 = cv.glm(df2, reg9_cv)$delta[1]
cvreg9 #0.5867

##Adding significant categorical variables
attach(df2)
df2$release_month = as.factor(df2$release_month)
df2$language = as.factor(df2$language)
df2$maturity_rating = as.factor(df2$maturity_rating)
df2$colour_film = as.factor(df2$colour_film)

#Adding maturity rating
reg10 = lm(imdb_score ~ maturity_rating + 
              bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
            + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
            + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
              bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) 
            + movie_budget + drama + biography + crime + comedy + horror + action + family + music + romance + adventure + animation + documentary,
            data = df2)
summary(reg10) #R-squared:  0.505,	Adjusted R-squared:  0.4943 p-value: < 2.2e-16 


reg10_cv = glm(imdb_score ~ maturity_rating + 
             bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
           + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
           + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
             bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) 
           + movie_budget + drama + biography + crime + comedy + horror + action + family + music + romance + adventure + animation + documentary,
           data = df2)

cvreg10 = cv.glm(df2, reg10_cv)$delta[1]
cvreg10 #0.5898


#Adding release month
reg11 = lm(imdb_score ~ release_month + maturity_rating + 
              bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
            + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
            + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
              bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) 
            + movie_budget + drama + biography + crime + comedy + horror + action + family + music + romance + adventure + animation + documentary,
            data = df2)
summary(reg11) #R-squared:  0.5077,	Adjusted R-squared:  0.4941 p-value: < 2.2e-16


reg11_cv = glm(imdb_score ~ release_month + maturity_rating + 
             bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
           + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
           + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
             bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) 
           + movie_budget + drama + biography + crime + comedy + horror + action + family + music + romance + adventure + animation + documentary,
           data = df2)

cvreg11 = cv.glm(df2, reg11_cv)$delta[1]
cvreg11 #0.5935
#Since our model Adjusted R-squared dropped, we will not use this

#Adding color films
reg12 = lm(imdb_score ~ colour_film  + maturity_rating + 
              bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
            + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
            + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
              bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) 
            + movie_budget + drama + biography + crime + comedy + horror + action + family + music + romance + adventure + animation + documentary,
            data = df2)
summary(reg12) #R-squared:  0.5083,	Adjusted R-squared:  0.4974 p-value: < 2.2e-16

reg12_cv = glm(imdb_score ~ colour_film  + maturity_rating + 
             bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
           + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
           + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
             bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) 
           + movie_budget + drama + biography + crime + comedy + horror + action + family + music + romance + adventure + animation + documentary,
           data = df2)

cvreg12 = cv.glm(df2, reg12_cv)$delta[1]
cvreg12 #0.5861


#Testing language and isEnglish to see which one is better
reg13a = lm(imdb_score ~ language + colour_film + maturity_rating + 
              bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
            + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
            + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
              bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) 
            + movie_budget + drama + biography + crime + comedy + horror + action + family + music + romance + adventure + animation + documentary,
            data = df2)
summary(reg13a) #R-squared:  0.5203,	Adjusted R-squared:  0.5052 p-value: < 2.2e-16

reg13b = lm(imdb_score ~ isEnglish + colour_film + maturity_rating + 
              bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
            + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
            + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
              bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) 
            + movie_budget + drama + biography + crime + comedy + horror + action + family + music + romance + adventure + animation + documentary,
            data = df2)
summary(reg13b) #R-squared:  0.5163,	Adjusted R-squared:  0.5054 p-value: < 2.2e-16

#Since isEnglish has a higher adjusted R-squared, we will use this for our model
reg13 = reg13b
summary(reg13)

reg13_cv = glm(imdb_score ~ isEnglish + colour_film + maturity_rating + 
              bs(duration, knots = c(k1_d,k2_d,k3_d), degree = 1) 
            + bs(movie_meter_IMDBpro, knots = c(k1_p, k2_p), degree = 2)
            + bs(nb_news_articles, knots = c(k1_a, k2_a), degree = 2) +
              bs(release_year,knots=c(k1_y, k2_y, k3_y),degree=1) 
            + movie_budget + drama + biography + crime + comedy + horror + action + family + music + romance + adventure + animation + documentary,
            data = df2)
cvreg13 = cv.glm(df2, reg13_cv)$delta[1]
cvreg13 #0.5753

predict(reg13, testFinal)


#Heteoskedacity
residualPlots(reg13)

ncvTest(reg13)

reg13_adj=coeftest(reg13, vcov = vcovHC(reg13, type = "HC1"))
reg13_adj

#### Generate Final Model in Table Format
install.packages("knitr")
install.packages("kableExtra")
library(knitr)
library(kableExtra)

# Extract coefficients, standard errors, and significance levels from reg13_adj
coefs <- reg13_adj[, 1]
robust_se <- reg13_adj[, 2]
p_values <- reg13_adj[, 4]
significance <- ifelse(p_values < 0.001, "***",
                       ifelse(p_values < 0.01, "**",
                              ifelse(p_values < 0.05, "*", "")))

# Create a data frame to match the desired layout
reg_table <- data.frame(
  Variable = rownames(reg13_adj),
  Coefficient = round(coefs, 2),
  `Standard Error` = paste0("(", round(robust_se, 2), ")"),
  Significance = significance
)

# Create the table
kable(reg_table, format = "html", col.names = c("Variable", "Coefficient", "Standard Error", "Significance")) %>%
  kable_styling(full_width = F, bootstrap_options = c("striped", "hover", "condensed")) %>%
  row_spec(0, bold = TRUE) # Make the header row bold


##################MISC MODEL ATTEMPTS##################
#######################################################


##PREVIOUS SPLINE MODEL FOR DURATION

#previous spline model for duration 
k1 = quantile(duration, 1/2)

#splines
spline1_duration_old=lm(imdb_score~ bs(duration,knots=c(k1),degree=1), data = df2)
spline2_duration_old=lm(imdb_score ~ bs(duration,knots=c(k1),degree=2), data = df2)
spline3_duration_old=lm(imdb_score ~ bs(duration,knots=c(k1),degree=3), data = df2)         
spline4_duration_old=lm(imdb_score ~ bs(duration,knots=c(k1),degree=4), data = df2)
spline5_duration_old=lm(imdb_score ~ bs(duration,knots=c(k1),degree=5), data = df2)


summary(spline1_duration_old) #RSE 0.9895, R^2: 0.1753, p < 2.2e-16
summary(spline2_duration_old) #RSE 0.9765  R^2 0.1973   adj R20.1953   | p value  2.2e-16
summary(spline3_duration_old) #RSE : 0.9836    R^2: 0.2021
summary(spline4_duration_old) #R^2 0.2002 -- R2 goes down
summary(spline4_duration_old) 
anova(spline1_duration_old,spline2_duration_old,spline3_duration_old,spline4_duration_old,spline5_duration_old)

#######################IGNORE ABOVE CODE BLOCK IF WE WANT##########
###################################################################
###Heat map to visualize selected R^2 value for each predictor
data <- data.frame(
  Variable = factor(c('Duration', 'IMDBPro meter', 'No. of News Articles', 'Release Year', 'Movie Budget', 'Number of Faces', 'Star1 Meter'), 
                    # Reverse the levels to get Duration at top and Movie Budget at bottom
                    levels = rev(c('Duration', 'IMDBPro meter', 'No. of News Articles', 'Release Year', 'Movie Budget', 'Number of Faces', 'Star1 Meter'))),
  Linear = c(0.1748, 0.008736, 0.09665, 0.03743, 0.005227, 0.0068, 0.0008514),
  Poly_2 = c(0.1945, 0.04172, 0.121, 0.04224, 0.006079, 0.00866, 0.001346),
  Poly_3 = c(0.1948, 0.07405, 0.1307, 0.04632, 0.006082, 0.008732, 0.001401),
  Spline_1 = c(0.2099, 0.1884, 0.1287, 0.04753, NA, 0.008838, 0.001726),
  Spline_2 = c(0.2104, 0.1955, 0.1339, 0.04632, NA, 0.008842, 0.001967),
  Spline_3 = c(0.2111, 0.1927, 0.1362, 0.04726, NA, 0.009489, 0.003937)
)

#install
library(reshape2)
library(ggplot2)

melted_data <- melt(data, id.vars = 'Variable')
melted_data$selected <- FALSE

# Mark selected models
selected_pairs <- data.frame(
  Variable = c('Duration', 'IMDBPro meter', 'No. of News Articles', 'Release Year', 'Movie Budget', 'Number of Faces', 'Star1 Meter'),
  Model = c('Spline_1', 'Spline_2', 'Spline_3', 'Spline_1', 'Linear', 'Spline_1', 'Spline_3')
)

# Mark selected models in the data
for(i in 1:nrow(selected_pairs)) {
  melted_data$selected[melted_data$Variable == selected_pairs$Variable[i] & 
                         melted_data$variable == selected_pairs$Model[i]] <- TRUE
}

# Create the heatmap
p <- ggplot(melted_data, aes(x = variable, y = Variable)) +
  geom_tile(aes(fill = value), color = "white") +
  geom_tile(data = subset(melted_data, selected), 
            fill = NA, 
            color = "yellow", 
            size = 1.5) +
  scale_fill_gradient(low = "#f2e6ff", high = "#4b0082", na.value = "grey90") +
  geom_text(aes(label = sprintf("%.3f", value)), color = ifelse(melted_data$value > 0.15, "white", "black"), size = 3.5) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    plot.title = element_text(size = 12, hjust = 0.5)
  ) +
  labs(title = "R² Values by Model Type and Variable", 
       fill = "R² Value",
       x = "Model Type",
       y = "Variable") +
  scale_x_discrete(labels = c('Linear', 'Poly, deg 2', 'Poly, deg 3', 
                              'Spline, deg 1', 'Spline, deg 2', 'Spline, deg 3'))

# Plot the heatmap
print(p)

