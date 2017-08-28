library(qvalue)
data(hedenfalk)
p <- hedenfalk$p
pi0<-pi0est(p)$pi0 # 0.669926

# qvalue validation
qv_q<-qvalue(p)
qv_q_pfdr<-qvalue(p, pfdr=TRUE)

qvalue_ref_data<-data.frame("p"=p, "q_default"=qv_q$qvalues, "q_pfdr"=qv_q_pfdr$qvalues)
write.csv(qvalue_ref_data, file="test_qvalue_ref_data.csv", row.names=FALSE)

# pi0est validation
nullRatio <- pi0est(p)$pi0
nullRatioS <- pi0est(p, lambda=seq(0.40, 0.95, 0.05), smooth.log.pi0="TRUE")$pi0
nullRatioM <- pi0est(p, pi0.method="bootstrap")$pi0

# lfdr validation
lfdr_default <- lfdr(p,pi0)
lfdr_monotone_false <- lfdr(p, pi0, monotone=FALSE)
lfdr_transf_logit <- lfdr(p, pi0, transf="logit")
lfdr_eps <- lfdr(p, pi0, eps=10^-2)

lfdr_ref_data<-data.frame("p"=p, "lfdr_default"=lfdr_default, "lfdr_monotone_false"=lfdr_monotone_false, "lfdr_transf_logit"=lfdr_transf_logit, "lfdr_eps"=lfdr_eps)
write.csv(lfdr_ref_data, file="test_lfdr_ref_data.csv", row.names=FALSE)
