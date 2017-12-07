module propagate
    use config
    use io
    use feature_util, only : int_in_intarray
    use util, only : allocate_dydx, zero_weights
    use init, only : allocate_weights

    implicit none

    external :: dgemv
    external :: dgemm
    external :: dcopy
    real(8),external :: ddot
            
    type(weights),public,allocatable :: d2ydxdw(:,:)
    real(8),public,allocatable :: sub_A1(:,:),sub_A2(:,:)
    real(8),public,allocatable :: sub_B(:,:),sub_C(:,:),sub_D(:,:)

    contains
        subroutine forward_propagate(conf,atm,set_type)
            implicit none

            integer,intent(in) :: conf,atm,set_type

            !* scratch
            integer :: nrow,ncol,ii
            real(8) :: x_atom(1:D+1)

    
            !* copy feature vector (inlcuding null for bias)
            call dcopy(D+1,data_sets(set_type)%configs(conf)%x(:,atm),1,x_atom,1)

            !------------------!
            !* hidden layer 1 *!
            !------------------!
            
            nrow = D  + 1
            ncol = net_dim%hl1

            !* x includes null dimension for bias
            call dgemv('t',nrow,ncol,1.0d0,net_weights%hl1(:,:),nrow,&
                    &x_atom,1,0.0d0,net_units%a%hl1,1)

            !* null value for bias
            net_units%z%hl1(1) = 1.0d0

            do ii=1,net_dim%hl1,1
                !* nonlinear map of linear models
                net_units%z%hl1(ii+1) = activation(net_units%a%hl1(ii))
            end do
            !------------------!
            !* hidden layer 2 *!
            !------------------!

            nrow = net_dim%hl1 + 1
            ncol = net_dim%hl2

            !* x includes null dimension for bias
            call dgemv('t',nrow,ncol,1.0d0,net_weights%hl2(:,:),nrow,net_units%z%hl1,&
                &1,0.0d0,net_units%a%hl2,1)

            !* null value for bias
            net_units%z%hl2(1) = 1.0d0

            do ii=1,net_dim%hl2,1
                !* nonlinear map of linear models
                net_units%z%hl2(ii+1) = activation(net_units%a%hl2(ii))
            end do

            !-------------------------!
            !* final layer to output *!
            !-------------------------!
            
            !* predicted energy
            data_sets(set_type)%configs(conf)%current_ei(atm) = ddot(net_dim%hl2+1,net_weights%hl3,1,&
                    &net_units%z%hl2,1)
            !* must compute forces seperately once have iterated over all atoms in conf 
        end subroutine

        subroutine backward_propagate(conf,atm,set_type)
            !===============================================!
            ! calculate dy/dw for all weights               !
            !===============================================!

            
            implicit none

            integer,intent(in) :: conf,atm,set_type

            !* scratch
            integer :: ii,jj,kk
            real(8) :: tmp1
            
            !-------------------!
            !* delta back prop *!
            !-------------------!

            !* layer 2 *!
            
            !* delta_i^(2) = h'(a_i^(2)) * w_i^(3) 
            do ii=1,net_dim%hl2,1
                !* activation derivatives
                net_units%a_deriv%hl2(ii) = activation_deriv(net_units%a%hl2(ii))
                !* bias does not have node
                net_units%delta%hl2(ii) = net_units%a_deriv%hl2(ii)*net_weights%hl3(ii+1)
            end do
            
            !* layer 1 *!
            
            !* delta_i^(1) = h'(a_i^(1)) * sum_j w_ij^(2) delta_i^(2)
            call dgemv('n',net_dim%hl1,net_dim%hl2,1.0d0,net_weights%hl2(2:net_dim%hl1+1,1:net_dim%hl2),&
                    &net_dim%hl1,net_units%delta%hl2,1,0.0d0,net_units%delta%hl1,1)
            
            do ii=1,net_dim%hl1,1
                !* activation derivatives
                net_units%a_deriv%hl1(ii) = activation_deriv(net_units%a%hl1(ii))

                net_units%delta%hl1(ii) = net_units%a_deriv%hl1(ii)*net_units%delta%hl1(ii)
            end do
            !* derivative of output wrt. weights *!

            !---------------!
            !* final layer *!
            !---------------!

            !* bias
            dydw%hl3(1) = 1.0d0

            do ii=1,net_dim%hl2,1
                dydw%hl3(ii+1) = net_units%z%hl2(ii+1)
            end do
            !----------------!
            !* second layer *!
            !----------------!
          
            !* bias
            do ii=1,net_dim%hl2,1
                dydw%hl2(1,ii) = net_units%a_deriv%hl2(ii)*net_weights%hl3(ii+1)
            end do

            call dgemm('n','n',net_dim%hl1,net_dim%hl2,1,1.0d0,net_units%z%hl1(2),&
                    &net_dim%hl1,net_units%delta%hl2,1,0.0d0,dydw%hl2(2:net_dim%hl1+1,1:net_dim%hl2),net_dim%hl1)
            !---------------!
            !* first layer *!
            !---------------!

            !* bias
            do ii=1,net_dim%hl1,1
                dydw%hl1(1,ii) = net_units%a_deriv%hl1(ii)*&
                        &sum(net_weights%hl2(ii+1,:)*net_units%delta%hl2)
            end do
            
            call dgemm('n','n',D,net_dim%hl1,1,1.0d0,data_sets(set_type)%configs(conf)%x(2:D+1,atm),&
                    &D,net_units%delta%hl1,1,0.0d0,dydw%hl1(2:D+1,1:net_dim%hl1),D)

            
            !---------------------------!
            !* derivative wrt features *!
            !---------------------------!
            
            dydx(:,atm) = 0.0d0
            
            do jj=1,net_dim%hl2
                do  ii=1,net_dim%hl1
                    tmp1 = net_units%delta%hl2(jj)*net_weights%hl2(ii+1,jj)*&
                            &net_units%a_deriv%hl1(ii)
                    do kk=1,D,1
                        dydx(kk,atm) = dydx(kk,atm) + tmp1*net_weights%hl1(kk+1,ii)
                    end do
                end do
            end do
            !call dgemm("n","n",D,net_dim%hl2,net_dim%hl1,1.0d0,&
            !        &net_weights%hl2(2:D+1,1:net_dim%net_dim%hl2),D,tmp_array,net_dim%hl1,dydx)

        end subroutine backward_propagate

        subroutine init_forceloss_subsidiary_mem()
            implicit none

            allocate(sub_A1(net_dim%hl1,D))
            allocate(sub_A2(net_dim%hl2,net_dim%hl1))
            allocate(sub_B(net_dim%hl1,D))
            allocate(sub_C(net_dim%hl2,net_dim%hl1))
            allocate(sub_D(D,net_dim%hl1))
        end subroutine init_forceloss_subsidiary_mem

        subroutine deallocate_forceloss_subsidiary_mem()
            implicit none
            deallocate(sub_A1)
            deallocate(sub_A2)
            deallocate(sub_B)
            deallocate(sub_C)
            deallocate(sub_D)
        end subroutine deallocate_forceloss_subsidiary_mem

        subroutine forceloss_weight_derivative_subsidiary()
            implicit none

            !* compute matrices A_1,A_2,B,C for conf,atm
    
            integer :: ii,jj
            
            do ii=1,net_dim%hl1
                do jj=1,net_dim%hl2
                    sub_A2(jj,ii) = net_units%a_deriv%hl2(jj)*net_weights%hl2(jj,ii)
                end do
            end do

            do ii=1,D
                do jj=1,net_dim%hl1
                    sub_A1(jj,ii) = net_units%a_deriv%hl1(jj)*net_weights%hl1(jj,ii)
                end do
            end do
        end subroutine forceloss_weight_derivative_subsidiary


        subroutine compute_forceloss_weight_derivatives()
            implicit none

            

        end subroutine compute_forceloss_weight_derivatives

        subroutine calculate_forces(set_type,conf)
            implicit none

            !* args
            integer,intent(in) :: set_type,conf

            !* scratch
            integer :: atm,natm,ii,jj,deriv_idx
           
            natm = data_sets(set_type)%configs(conf)%n

            data_sets(set_type)%configs(conf)%current_fi = 0.0d0
    
            do atm=1,natm,1
                do ii=1,D,1
                    if (data_sets(set_type)%configs(conf)%x_deriv(ii,atm)%n.eq.0) then
                        !* feature doest not contain any position info
                        cycle
                    end if
                    
                    do deriv_idx=1,data_sets(set_type)%configs(conf)%x_deriv(ii,atm)%n,1
                        !* d feature_{ii,atm} / d r_jj
                        jj = data_sets(set_type)%configs(conf)%x_deriv(ii,atm)%idx(deriv_idx)
       
                         !* -= d E_atm / d feature_{ii,atm} * d feature_{ii,atm} / d r_jj
                        data_sets(set_type)%configs(conf)%current_fi(:,jj) = &
                                &data_sets(set_type)%configs(conf)%current_fi(:,jj) - dydx(ii,atm) * &
                                &data_sets(set_type)%configs(conf)%x_deriv(ii,atm)%vec(:,deriv_idx)
                    end do !* end loop over atoms jj contributing to feature (ii,atm)
                
                end do !* end loop over features ii
            end do !* end loop over local cell atoms
        end subroutine calculate_forces

!        subroutine backprop_all_forces(set_type)
!            !===================================================!
!            ! Assumes that net has been forward propagated on   !
!            ! all configurations in set_type                    !
!            !                                                   !
!            ! Parameters                                        !
!            ! ----------                                        !
!            ! set_type : int, allowed values = 1,2              !
!            !     The data set to compute forces for            !  
!            !===================================================!
!            
!            implicit none
!
!            !* args
!            integer,intent(in) :: set_type
!
!            !* scratch
!            integer :: atm,conf,ii
!            type(weights),allocatable :: d2ydxdw(:,:)
!        
!            call init_forceloss_subsidiary_mem()
!            call forceloss_weight_derivative_subsidiary()
!            
!            do conf=1,data_sets(set_type)%nconf,1
!                if (allocated(dydx)) then
!                    deallocate(dydx)
!                end if
!                call allocate_dydx(set_type,conf)
!
!                if (allocated(d2ydxdw)) then
!                    deallocate(d2ydxdw)
!                end if
!                allocate(d2ydxdw(data_sets(set_type)%configs(conf)%n,D)) 
!                do ii=1,D
!                    do atm=1,data_sets(set_type)%configs(conf)%n
!                        call allocate_weights(d2ydxdw(atm,ii))
!                        call zero_weights(d2ydxdw(atm,ii))
!                    end do
!                end do
!
!                do atm=1,data_sets(set_type)%configs(conf)%n,1
!                    call backward_propagate(conf,atm,set_type)
!                end do 
!                call calculate_forces(set_type,conf)
!
!                deallocate(dydx)
!            end do
!        end subroutine backprop_all_forces

        real(8) function activation(ain)
            implicit none

            real(8),intent(in) :: ain

            if (nlf.eq.1) then
                activation = logistic(ain)
            else if (nlf.eq.2) then
                activation = tanh(ain)
            else
                call error("activation","unsupported nonlinear function")
                call exit(0)
            end if
        end function activation

        real(8) function activation_deriv(ain)
            implicit none

            real(8),intent(in) :: ain

            if (nlf.eq.1) then
                activation_deriv = logistic_deriv(ain)
            else if (nlf.eq.2) then
                activation_deriv = tanh_deriv(ain)
            else
                call error("activation_deriv","unsupported nonlinear function")
                call exit(0)
            end if
        end function activation_deriv

        real(8) function logistic(x)
            real(8),intent(in) :: x
            logistic = 1.0d0/(1.0d0 + exp(-x))
        end function logistic

        real(8) function logistic_deriv(x)
            !* compute using logarithm to avoid 1/0 for large x
            
            real(8),intent(in) :: x

            !* scratch
            real(8) :: tmp
            tmp = exp(-x)
            logistic_deriv = exp(-(x+2.0d0*log(tmp+1.0d0)) )
        end function logistic_deriv

        real(8) function tanh_deriv(x)
            real(8),intent(in) :: x
            !tanh_deriv = 1.0d0 - (tanh(x)**2)
            tanh_deriv = (1.0d0/cosh(x))**2
        end function tanh_deriv
end module propagate
