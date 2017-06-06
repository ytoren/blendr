#' Mesh ML
#'
#' The function \code{meshML} designed to mesh (or embed) the score calculated by a machine learning (ML) method into a parametric regression, in particular a generalized linear model (GLM). 
#' The motivation is to harness the power of ML methods where a simple interpretation of the marginal effect of some variables is not mandatory, while keeping a transparent structure with simple interpretation (and direct understanding of marginal effects) for variables of interest.  \cr \cr
#' Since we expect possible co-linearities to exist between the variables used by the 2 methods, and since the parameters of interest are the ones included in the parametric regression, the ML regression will run on an orthogonalized version of the dependent variables (this will be done using simple linear regression and regression on the residuals) \cr \cr
#' @param data The data-frame containing the data
#' @param p_FUN A function to calculate the blended parametric regression. Defaults to R's \code{glm}
#' @param p_formula A two-sided formula with the dependent variable and explaining variables to be included directly in the parametric regression
#' @param p_args = A named list of arguments to be passed to the parametric regression function (e.g list(familiy = binomial) for logistic regression)
#' @param ml_FUN A function to calculate the (non-parametric) ML regression. When set to \code{NULL} (the default) the will try to load and use \code{randomForest}.
#' @param ml_formula A one-sided formula with dependent variables to be included in the ML regression (dependent variable is table form the parametric formula)
#' @param p_args = A named list of arguments to be passed to the ML regression function (e.g \code{list(familiy = binomial)} for logistic regression)
#' @return The function \code{blendML} returns an S3 object with class "blendML" that contains:
#' \item{ort}{A list containig the outputs of the orthogonalization process (a list of \code{"lm"} objects)}
#' \item{ML}{The optput of the non-parametric regression (by default an object of class \code{"randomForest"})}
#' \item{param}{The optput of the parametric regression (by default an object of class \code{"glm"})}
#' @keywords GLM, MachineLearning, blending
#' @export
#' @examples
#' # Linear regression & random forests
#' x <- blendML(data = mtcars, p_formula = mpg ~ disp, ml_formula =  ~ cyl + hp + wt)
#' summary(x$param)
#' # Logistic regression & random forests
#' x <- blendML(data = mtcars, p_formula = vs ~ disp, p_args = list(family = binomial), ml_formula =  ~ cyl + hp + wt)
#' summary(x$param)

  
meshML <- function(
  data, 
  p_FUN = glm,
  p_formula, 
  p_args = list(family = gaussian),
  ml_FUN = NULL,
  ml_formula, 
  ml_args = list()
  ) 
{
  require(magrittr)
  require(purrr)
  require(dplyr)
  result <- NULL

  # Load libraries for supported back-ends
  if(is.null(ml_FUN)) {
    if(installed.packages() %>% 
       rownames() %>% 
       grep(pattern = 'randomForest') 
       %>% length() > 0) {
      cat('Defaulting to Random Forests \n')
      require(randomForest)
      ml_FUN = randomForest
    } else {stop('please install the package \"randomForest\" first.')}
  }

  ## Step 1: regress each component of ml_formula against the RHS of glm_formula
  attr(terms(ml_formula), 'term.labels') %>% 
    paste0(' ~ .') %>% 
    map(as.formula) %>% 
    map(function(y) {update(y, terms(p_formula))}) %>% 
    map(function(y) {lm(y, data = data)}) -> result$ort
  
  ## Step 2: run a random forest model on the residuals
  result$ml_residual_models %>%
    map(function(y) {residuals(y)}) %>%
    setNames(attr(terms(ml_formula), 'term.labels')) %>% 
    as.data.frame() %>%
    list() %>% 
    setNames('x') %>% 
    append(ml_args) %>% 
    append(list(y = with(data, get(as.character(attr(terms(p_formula), 'variables')[[2]]))))) %>% 
    do.call(what = ml_FUN) -> result$ML
  
  ## Step 3: embed the ML score into a GLM
  model.frame(formula = p_formula, data = data) %>% 
    mutate(ML_score = predict(result$rf)) %>% 
    list() %>% 
    setNames('data') %>%
    append(p_args) %>% 
    append(list(formula = p_formula)) %>% 
    do.call(what = p_FUN) -> result$param
  
  class(result) <- 'blendML'
  
  return(result)
}