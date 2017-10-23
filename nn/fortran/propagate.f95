module propagate
    use config
    use io

    implicit none

    external :: dgemv
    external :: dgemm
    real(8),external :: ddot

    contains
        subroutine forward_propagate(conf,atm,set_type)
            implicit none

            integer,intent(in) :: conf,atm,set_type

            !* scratch
            integer :: nrow,ncol,ii

            !------------------!
            !* hidden layer 1 *!
            !------------------!
            
            nrow = D  + 1
            ncol = net_dim%hl1

            !* x includes null dimension for bias
            call dgemv('t',nrow,ncol,1.0d0,net_weights%hl1(:,:),nrow,&
                    &data_sets(set_type)%configs(conf)%x(:,atm),1,0.0d0,net_units%a%hl1,1)

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
        
            !* predicted force
            data_sets(set_type)%configs(conf)%current_fi(1:3,atm) = 0.0d0

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
                !* bias does not have node
                net_units%delta%hl2(ii) = activation_deriv(net_units%a%hl2(ii))*&
                        &net_weights%hl3(ii+1)
            end do
            
            !* layer 1 *!
            
            !* delta_i^(1) = h'(a_i^(1)) * sum_j w_ij^(2) delta_i^(2)
            call dgemv('n',net_dim%hl1,net_dim%hl2,1.0d0,net_weights%hl2(2:net_dim%hl1+1,1:net_dim%hl2),&
                    &net_dim%hl1,net_units%delta%hl2,1,0.0d0,net_units%delta%hl1,1)
            do ii=1,net_dim%hl1,1
                net_units%delta%hl1(ii) = activation_deriv(net_units%a%hl1(ii))*&
                        &net_units%delta%hl1(ii)
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
                dydw%hl2(1,ii) = activation_deriv(net_units%a%hl2(ii))*net_weights%hl3(ii+1)
            end do

            call dgemm('n','n',net_dim%hl1,net_dim%hl2,1,1.0d0,net_units%z%hl1(2),&
                    &net_dim%hl1,net_units%delta%hl2,1,0.0d0,dydw%hl2(2:net_dim%hl1+1,1:net_dim%hl2),net_dim%hl1)
            !---------------!
            !* first layer *!
            !---------------!

            !* bias
            do ii=1,net_dim%hl1,1
                dydw%hl1(1,ii) = activation_deriv(net_units%a%hl1(ii))*&
                        &sum(net_weights%hl2(ii+1,:)*net_units%delta%hl2)
            end do
            
            call dgemm('n','n',D,net_dim%hl1,1,1.0d0,data_sets(set_type)%configs(conf)%x(2:D+1,atm),&
                    &D,net_units%delta%hl1,1,0.0d0,dydw%hl1(2:D+1,1:net_dim%hl1),D)

            !---------------------------!
            !* derivative wrt features *!
            !---------------------------!

            dydx = 0.0d0

            do jj=1,net_dim%hl2
                tmp1 = net_weights%hl3(jj+1)*activation_deriv(net_units%a%hl2(jj))
                do  ii=1,net_dim%hl1
                    do kk=1,D,1
                        dydx(kk) = dydx(kk) + tmp1*net_weights%hl2(ii+1,jj)*&
                                &activation_deriv(net_units%a%hl1(ii))*net_weights%hl1(kk+1,ii)
                    end do
                end do
            end do

        end subroutine backward_propagate

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
            real(8),intent(in) :: x

            !* scratch
            real(8) :: tmp

            tmp = exp(-x)
            logistic_deriv = tmp/((1.0d0+tmp)**2)
        end function logistic_deriv

        real(8) function tanh_deriv(x)
            real(8),intent(in) :: x
            tanh_deriv = 1.0d0 - (tanh(x)**2)
        end function tanh_deriv
end module propagate
